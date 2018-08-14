module IKMimic

using RigidBodyDynamics
using RigidBodyDynamics.Graphs: target, TreePath
import RigidBodyDynamics.CustomCollections
using RigidBodyDynamics.PDControl: group_error
using Rotations
using Compat

const VectorView{T} = SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
const MatrixView{T} = SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int}, UnitRange{Int}}, false}

#=
Given the current state of the reference model, x1, find an associated
state of the target model, x2, to match the position of a collection of
bodies and the orientation of another collection of bodies between the
two models.

Currently assumes that the origins and orientations of the matching bodies
are identical in the two models.

To do that, we create a function y = f(x), such that y is the concatenation
of the positions, velocities, orientations, and twists of the target bodies.
We then linearize f: y ≈ f(x0) + J (x - x0) = y0 + J (x - x0)

We now have an optimization problem:

minimize ||J (x2 - x20) + y20 - y10 ||^2
  x2

= ||Jx2 + (y20 - y10 - Jx20) || ^2
= ||Jx2 + b||^2
    where b = (y20 - y10 - J x20)
= x2'J'Jx2 + 2b'Jx2 + b'b

which we can solve exactly by setting its gradient to zero:

J'Jx2 + J'b = 0

x2 = (J'J) \ -J'b

alternatively, we can perform a gradient descent update:

x2 = x2 - α * (J'Jx2 + J'b)

=#

struct PointTask{T}
    path_to_body::TreePath{RigidBody{T}, Joint{T}}
    jac::Spatial.PointJacobian{Matrix{T}}
    J_q::MatrixView{T}
    J_v::MatrixView{T}
end

struct PointError{T}
    reference_body::RigidBody{T}
    result_body::RigidBody{T}
    e_q::VectorView{T}
    e_v::VectorView{T}
end

function update!(task::PointTask, state::MechanismState, J_qdot_to_v)
    body = target(task.path_to_body)
    p = transform_to_root(state, body) * Point3D(default_frame(body), 0., 0, 0)
    point_jacobian!(task.jac, state, task.path_to_body, p)
    task.J_v .= task.jac.J

    # task.J_q .= task.jac.J * parent(J_qdot_to_v)
    A_mul_B!(task.J_q, task.jac.J, parent(J_qdot_to_v))
    nothing
end

function update!(e::PointError, reference_state::MechanismState, result_state::MechanismState)
    H1 = transform_to_root(reference_state, e.reference_body)
    H2 = transform_to_root(result_state, e.result_body)
    T1 = twist_wrt_world(reference_state, e.reference_body)
    T2 = twist_wrt_world(result_state, e.result_body)
    e.e_q .= group_error(translation(H2), translation(H1))
    ṗ1 = point_velocity(T1, H1 * Point3D(default_frame(e.reference_body), 0., 0, 0))
    @framecheck ṗ1.frame root_frame(reference_state.mechanism)
    ṗ2 = point_velocity(T2, H2 * Point3D(default_frame(e.result_body), 0., 0, 0))
    @framecheck ṗ2.frame root_frame(result_state.mechanism)
    e.e_v .= group_error(ṗ2.v, ṗ1.v)
    nothing
end

struct OrientationTask{T}
    path_to_body::TreePath{RigidBody{T}, Joint{T}}
    jac::Spatial.GeometricJacobian{Matrix{T}}
    J_q::MatrixView{T}
    J_v::MatrixView{T}
end

struct OrientationError{T}
    reference_body::RigidBody{T}
    result_body::RigidBody{T}
    e_q::VectorView{T}
    e_v::VectorView{T}
end

function update!(task::OrientationTask, state::MechanismState, J_qdot_to_v)
    body = target(task.path_to_body)
    geometric_jacobian!(task.jac, state, task.path_to_body)
    task.J_v .= angular(task.jac)

    # task.J_q .= angular(task.jac) * J_qdot_to_v
    A_mul_B!(task.J_q, angular(task.jac), parent(J_qdot_to_v))
    nothing
end

function update!(e::OrientationError, reference_state::MechanismState, result_state::MechanismState)
    H1 = transform_to_root(reference_state, e.reference_body)
    H2 = transform_to_root(result_state, e.result_body)
    T1 = twist_wrt_world(reference_state, e.reference_body)
    T2 = twist_wrt_world(result_state, e.result_body)
    rot = RodriguesVec(group_error(rotation(H2), rotation(H1)))
    e.e_q .= (rot.sx, rot.sy, rot.sz)
    e.e_v .= group_error(angular(T2), angular(T1))
    nothing
end

struct TaskMap{T}
    point_tasks::Vector{PointTask{T}}
    point_errors::Vector{PointError{T}}
    orientation_tasks::Vector{OrientationTask{T}}
    orientation_errors::Vector{OrientationError{T}}
    J::Matrix{T}
    y2_minus_y1::Vector{T}
    J_qdot_to_v::CustomCollections.SegmentedBlockDiagonalMatrix{T, Matrix{T}}

    function TaskMap{T}(result_mechanism::Mechanism{T}, point_matches::AbstractVector{Tuple{RigidBody{T}, RigidBody{T}}}, orientation_matches::AbstractVector{Tuple{RigidBody{T}, RigidBody{T}}}) where {T}
        nq = num_positions(result_mechanism)
        nv = num_velocities(result_mechanism)
        ny = 6 * (length(point_matches) + length(orientation_matches))
        J = zeros(ny, nq + nv)
        y2_minus_y1 = zeros(ny)

        point_tasks = PointTask{T}[]
        point_errors = PointError{T}[]
        row = 1
        for (reference_body, result_body) in point_matches
            path_to_body = path(result_mechanism, root_body(result_mechanism), result_body)
            I = row - 1 + (1:6)
            J_q = view(J, I[1:3], 1:nq)
            J_v = view(J, I[4:6], nq + (1:nv))  # TODO: missing ∂ṗ/∂q
            jac = PointJacobian(zeros(3, nv), root_frame(result_mechanism))
            task = PointTask{T}(path_to_body, jac, J_q, J_v)
            e_q = view(y2_minus_y1, I[1:3])
            e_v = view(y2_minus_y1, I[4:6])
            err = PointError{T}(reference_body, result_body, e_q, e_v)
            push!(point_tasks, task)
            push!(point_errors, err)
            row += 6
        end
        orientation_tasks = OrientationTask{T}[]
        orientation_errors = OrientationError{T}[]
        for (reference_body, result_body) in orientation_matches
            path_to_body = path(result_mechanism, root_body(result_mechanism), result_body)
            I = row - 1 + (1:6)
            J_q = view(J, I[1:3], 1:nq)
            J_v = view(J, I[4:6], nq + (1:nv))  # TODO: missing ∂ω/∂q
            jac = GeometricJacobian(default_frame(result_body), root_frame(result_mechanism), root_frame(result_mechanism), zeros(3, nv), zeros(3, nv))
            task = OrientationTask{T}(path_to_body, jac, J_q, J_v)
            e_q = view(y2_minus_y1, I[1:3])
            e_v = view(y2_minus_y1, I[4:6])
            err = OrientationError{T}(reference_body, result_body, e_q, e_v)
            push!(orientation_tasks, task)
            push!(orientation_errors, err)
            row += 6
        end

        # TODO: this jacobian could be allocated without actually constructing
        # a MechanismState
        J_qdot_to_v = RigidBodyDynamics.configuration_derivative_to_velocity_jacobian(MechanismState(result_mechanism))

        new{T}(point_tasks, point_errors, orientation_tasks, orientation_errors, J, y2_minus_y1, J_qdot_to_v)
    end
end

function TaskMap(reference_mechanism::Mechanism{T}, result_mechanism::Mechanism{T}, point_matches::AbstractVector{<:AbstractString}, orientation_matches::AbstractVector{<:AbstractString}) where {T}
    TaskMap{T}(result_mechanism,
        [(findbody(reference_mechanism, b), findbody(result_mechanism, b)) for b in point_matches],
        [(findbody(reference_mechanism, b), findbody(result_mechanism, b)) for b in orientation_matches])
end

function TaskMap(reference_mechanism::Mechanism, result_mechanism::Mechanism, point_and_orientation_matches::AbstractVector)
    TaskMap(reference_mechanism, result_mechanism, point_and_orientation_matches, point_and_orientation_matches)
end

function update!(task_map::TaskMap, reference_state::MechanismState, result_state::MechanismState)
    J_qdot_to_v = task_map.J_qdot_to_v
    RigidBodyDynamics.configuration_derivative_to_velocity_jacobian!(J_qdot_to_v, result_state)
    for task in task_map.point_tasks
        update!(task, result_state, J_qdot_to_v)
    end
    for task in task_map.orientation_tasks
        update!(task, result_state, J_qdot_to_v)
    end
    for e in task_map.point_errors
        update!(e, reference_state, result_state)
    end
    for e in task_map.orientation_errors
        update!(e, reference_state, result_state)
    end
end


struct IKMimicWorkspace{T}
    task_map::TaskMap{T}
    A::Matrix{T}
    b::Vector{T}
    A_fact::Base.RefValue{LinAlg.Cholesky{T, Matrix{T}}}
    g::Vector{T}
    x1::Vector{T}
    q1::VectorView{T}
    v1::VectorView{T}
    x2::Vector{T}
    q2::VectorView{T}
    v2::VectorView{T}

    function IKMimicWorkspace{T}(reference_mechanism::Mechanism{T}, result_mechanism::Mechanism{T}, task_map::TaskMap{T}) where {T}
        x1 = zeros(num_positions(reference_mechanism) + num_velocities(reference_mechanism))
        q1 = view(x1, 1:num_positions(reference_mechanism))
        v1 = view(x1, num_positions(reference_mechanism) + (1:num_velocities(reference_mechanism)))
        x2 = zeros(num_positions(result_mechanism) + num_velocities(result_mechanism))
        q2 = view(x2, 1:num_positions(result_mechanism))
        v2 = view(x2, num_positions(result_mechanism) + (1:num_velocities(result_mechanism)))

        # Need to temporarily make A pos def so we can easily allocate its
        # cholfact, so we initialize it to I
        A = Matrix(1.0I, length(x2), length(x2))
        b = fill(0.0, length(task_map.y2_minus_y1))
        A_fact = cholfact!(A)
        g = fill(0.0, length(x2))
        new{T}(task_map, A, b, Ref(A_fact), g, x1, q1, v1, x2, q2, v2)
    end
end

function IKMimicWorkspace(reference_mechanism::Mechanism{T}, result_mechanism::Mechanism{T}, args...) where {T}
    task_map = TaskMap(reference_mechanism, result_mechanism, args...)
    IKMimicWorkspace{T}(reference_mechanism, result_mechanism, task_map)
end

IKMimicWorkspace(x1::MechanismState, x2::MechanismState, args...) =
    IKMimicWorkspace(x1.mechanism, x2.mechanism, args...)

function clamp_to_position_bounds!(q::AbstractVector, joints::AbstractVector{<:Joint})
    i = 1
    for joint in joints
        for bound in position_bounds(joint)
            q[i] = clamp(q[i], bound)
            i += 1
        end
    end
    @assert i == length(q) + 1
end

function update!(work::IKMimicWorkspace, reference_state::MechanismState, result_state::MechanismState)
    update!(work.task_map, reference_state, result_state)
    copy!(work.q1, configuration(reference_state))
    copy!(work.v1, velocity(reference_state))
    copy!(work.q2, configuration(result_state))
    copy!(work.v2, velocity(result_state))
    J = work.task_map.J

    # A = J' * J
    At_mul_B!(work.A, J, J)

    # b = work.task_map.y2_minus_y1 - J * work.x2
    A_mul_B!(work.b, J, work.x2)
    work.b .= work.task_map.y2_minus_y1 .- work.b

    # work.x2 .= A \ (-J' * b)
    At_mul_B!(work.g, J, work.b)
    work.g .*= -1
    work.A_fact[] = cholfact!(Hermitian(work.A))
    A_ldiv_B!(work.x2, work.A_fact[], work.g)

    clamp_to_position_bounds!(work.q2, tree_joints(result_state.mechanism))
end

function ik_mimic!(result_state::MechanismState, reference_state::MechanismState, work::IKMimicWorkspace)
    update!(work, reference_state, result_state)
    set_configuration!(result_state, work.q2)
    set_velocity!(result_state, work.v2)
    nothing
end

function ik_mimic!(result_state, reference_state, args...)
    work = IKMimicWorkspace(reference_state.mechanism, result_state.mechanism, args...)
    ik_mimic!(result_state, reference_state, work)
end

# struct BodyData{T}
#     body::RigidBody{T}
#     path_to_body::TreePath{RigidBody{T}, Joint{T}}
#     J_geometric::Spatial.GeometricJacobian{Matrix{T}}
#     A_q_angular::MatrixView{T}
#     b_q_angular::VectorView{T}
#     A_q_linear::MatrixView{T}
#     b_q_linear::VectorView{T}
#     A_v_angular::MatrixView{T}
#     b_v_angular::VectorView{T}
#     A_v_linear::MatrixView{T}
#     b_v_linear::VectorView{T}
# end


# struct BodyMap{T}
#     body_data::Vector{BodyData{T}}
#     J_q_to_v::CustomCollections.SegmentedBlockDiagonalMatrix{T, Matrix{T}}
#     A::Matrix{T}
#     b::Vector{T}
#     x0::Vector{T}
#     b0::Vector{T}

#     function BodyMap(state::MechanismState{T}, bodynames::AbstractVector{<:AbstractString}) where {T}
#         mechanism = state.mechanism
#         nq = num_positions(state)
#         nv = num_velocities(state)
#         A = zeros(12 * length(bodynames), nq + nv)
#         b = zeros(12 * length(bodynames))
#         x0 = zeros(nq + nv)
#         b0 = similar(b)
#         J_q_to_v = RigidBodyDynamics.configuration_derivative_to_velocity_jacobian(state)
#         row = 1
#         bodydata = Vector{BodyData{T}}()
#         for bodyname in bodynames
#             body = findbody(mechanism, bodyname)
#             path_to_body = path(mechanism, root_body(mechanism), body)
#             J_geometric = geometric_jacobian(state, path_to_body)
#             rows = row:(row + 11)
#             A_q_angular = view(A, rows[1:3], 1:nq)
#             b_q_angular = view(b, rows[1:3])
#             A_q_linear = view(A, rows[4:6], 1:nq)
#             b_q_linear = view(b, rows[4:6])
#             A_v_angular = view(A, rows[7:9], nq + (1:nv))
#             b_v_angular = view(b, rows[7:9])
#             A_v_linear = view(A, rows[10:12], nq + (1:nv))
#             b_v_linear = view(b, rows[10:12])
#             push!(bodydata,
#                 BodyData{T}(body,
#                     path_to_body,
#                     J_geometric,
#                     A_q_angular,
#                     b_q_angular,
#                     A_q_linear,
#                     b_q_linear,
#                     A_v_angular,
#                     b_v_angular,
#                     A_v_linear,
#                     b_v_linear,
#                ))
#             row += 12
#         end
#         new{T}(bodydata, J_q_to_v, A, b, x0, b0)
#     end
# end

# function update_linearization!(data::BodyData, state::MechanismState, J_q_to_v)
#     geometric_jacobian!(data.J_geometric, state, data.path_to_body)
#     A_mul_B!(data.A_q_angular, angular(data.J_geometric), parent(J_q_to_v))
#     A_mul_B!(data.A_q_linear, linear(data.J_geometric), parent(J_q_to_v))
#     data.A_v_angular .= angular(data.J_geometric)
#     data.A_v_linear .= linear(data.J_geometric)
#     H = transform_to_root(state, data.body)
#     r = RodriguesVec(rotation(H))
#     data.b_q_angular .= (r.sx, r.sy, r.sz)
#     data.b_q_linear .= translation(H)
#     T = twist_wrt_world(state, data.body)
#     data.b_v_angular .= angular(T)
#     data.b_v_linear .= linear(T)
#     nothing
# end

# function update_linearization!(bodymap::BodyMap, state::MechanismState)
#     RigidBodyDynamics.configuration_derivative_to_velocity_jacobian!(bodymap.J_q_to_v, state)
#     for data in bodymap.body_data
#         update_linearization!(data, state, bodymap.J_q_to_v)
#     end
#     copy!(bodymap.x0, configuration(state))
#     copy!(bodymap.x0, CartesianRange((num_positions(state)+(1:num_velocities(state)),)), velocity(state), CartesianRange((1:num_velocities(state),)))
#     A_mul_B!(bodymap.b0, bodymap.A, bodymap.x0)
#     bodymap.b .= bodymap.b .- bodymap.b0;
#     nothing
# end

# _unwrap(x::Real, y::Real) = x + (mod((y - x) + π, 2π) - π)

# function _unwrap!(dest::BodyData, src::BodyData)
#     dest.b_q_angular .= _unwrap.(src.b_q_angular, dest.b_q_angular)
#     dest.b_v_angular .= _unwrap.(src.b_v_angular, dest.b_v_angular)
# end

# function _unwrap!(dest::BodyMap, src::BodyMap)
#     for i in eachindex(dest.body_data)
#         _unwrap!(dest.body_data[i], src.body_data[i])
#     end
# end

# mutable struct IKMimicWorkspace{T}
#     reference::BodyMap{T}
#     result::BodyMap{T}
#     H::Matrix{T}
#     H_fact::LinAlg.Cholesky{T, Matrix{T}}
#     p1::Vector{T}
#     f1::Vector{T}
#     f2::Vector{T}
#     x2::Vector{T}

#     function IKMimicWorkspace(reference::BodyMap{T}, result::BodyMap{T}) where {T}
#         H = eye(T, size(result.A, 2), size(result.A, 2))
#         H_fact = cholfact!(H)
#         p1 = zeros(T, size(reference.A, 1))
#         f1 = zeros(T, size(result.A, 2))
#         f2 = zeros(T, size(result.A, 2))
#         x2 = zeros(T, size(result.A, 2))
#         new{T}(reference, result, H, H_fact, p1, f1, f2, x2)
#     end
# end

# function IKMimicWorkspace(reference::MechanismState, result::MechanismState, bodynames::AbstractVector{<:AbstractString})
#     IKMimicWorkspace(BodyMap(reference, bodynames), BodyMap(result, bodynames))
# end

# struct SensitivityWorkspace{T}
#     A2tA1::Matrix{T}
#     J::Matrix{T}

#     function SensitivityWorkspace{T}(reference_nx::Integer, result_nx::Integer) where {T}
#         new{T}(zeros(T, result_nx, reference_nx),
#                zeros(T, result_nx, reference_nx))
#     end

# end


# function mimic_update!(work::IKMimicWorkspace)
#     _unwrap!(work.reference, work.result)
#     A1 = work.reference.A
#     b1 = work.reference.b
#     A2 = work.result.A
#     b2 = work.result.b

#     A_mul_B!(work.p1, A1, work.reference.x0)
#     work.p1 .= work.p1 .+ b1

#     At_mul_B!(work.H, A2, A2)

#     At_mul_B!(work.f1, A2, work.p1)
#     At_mul_B!(work.f2, A2, b2)
#     work.f1 .= work.f1 .- work.f2
#     work.H_fact = cholfact!(Hermitian(work.H))
#     A_ldiv_B!(work.x2, work.H_fact, work.f1)

# end

# function ik_mimic!(result::MechanismState, reference::MechanismState, work::IKMimicWorkspace, iters=1)
#     update_linearization!(work.reference, reference)
#     for i in 1:iters
#         update_linearization!(work.result, result)
#         mimic_update!(work)
#         nq = num_positions(result)
#         nv = num_velocities(result)
#         set_configuration!(result, work.x2[1:nq])
#         set_velocity!(result, work.x2[1 + (1:nv)])
#         # copy!(result, work.x2)
#     end
#     nothing
# end

# function ik_mimic!(result::MechanismState, reference::MechanismState, bodynames::AbstractVector{<:AbstractString}, iters=1)
#     work = IKMimicWorkspace(BodyMap(reference, bodynames),
#                             BodyMap(result, bodynames))
#     ik_mimic!(result, reference, work, iters)
# end

# function sensitivity(work::IKMimicWorkspace{T}) where {T}
#     sensitivity!(
#         SensitivityWorkspace{T}(
#             size(work.reference.A, 2), size(work.result.A, 2)),
#         work)
# end

# function sensitivity!(sensitivity_work::SensitivityWorkspace,
#                       work::IKMimicWorkspace)
#     At_mul_B!(sensitivity_work.A2tA1, work.result.A, work.reference.A)
#     A_ldiv_B!(sensitivity_work.J, work.H_fact, sensitivity_work.A2tA1)
#     sensitivity_work.J
# end


# (A2' * A2) \ (A2' * (A1 * x + b1) - A2' * b2)
# (A2' * A2) \ (A2' * A1) * x + (const)


# function mimic_update!(variable::BodyMap, reference::BodyMap)
#     _unwrap!(reference, variable)
#     p1 = reference.A * reference.x0 + reference.b
#     A2 = variable.A
#     b2 = variable.b
#     x2 = (A2' * A2) \ (A2' * p1 - A2' * b2)
#     copy!(variable.state, x2)
# end

end # module
