module IKMimic

using RigidBodyDynamics
import RigidBodyDynamics.Graphs
import RigidBodyDynamics.CustomCollections
using Rotations

const VectorView{T} = SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
const MatrixView{T} = SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int}, UnitRange{Int}}, false}

struct BodyData{T}
    body::RigidBody{T}
    path_to_body::Graphs.TreePath{RigidBody{T}, Joint{T}}
    J_geometric::Spatial.GeometricJacobian{Matrix{T}}
    A_q_angular::MatrixView{T}
    b_q_angular::VectorView{T}
    A_q_linear::MatrixView{T}
    b_q_linear::VectorView{T}
    A_v_angular::MatrixView{T}
    b_v_angular::VectorView{T}
    A_v_linear::MatrixView{T}
    b_v_linear::VectorView{T}
end


struct BodyMap{T}
    body_data::Vector{BodyData{T}}
    J_q_to_v::CustomCollections.SegmentedBlockDiagonalMatrix{T, Matrix{T}}
    A::Matrix{T}
    b::Vector{T}
    x0::Vector{T}
    b0::Vector{T}

    function BodyMap(state::MechanismState{T}, bodynames::AbstractVector{<:AbstractString}) where {T}
        mechanism = state.mechanism
        nq = num_positions(state)
        nv = num_velocities(state)
        A = zeros(12 * length(bodynames), nq + nv)
        b = zeros(12 * length(bodynames))
        x0 = zeros(nq + nv)
        b0 = similar(b)
        J_q_to_v = RigidBodyDynamics.configuration_derivative_to_velocity_jacobian(state)
        row = 1
        bodydata = Vector{BodyData{T}}()
        for bodyname in bodynames
            body = findbody(mechanism, bodyname)
            path_to_body = path(mechanism, root_body(mechanism), body)
            J_geometric = geometric_jacobian(state, path_to_body)
            rows = row:(row + 11)
            A_q_angular = view(A, rows[1:3], 1:nq)
            b_q_angular = view(b, rows[1:3])
            A_q_linear = view(A, rows[4:6], 1:nq)
            b_q_linear = view(b, rows[4:6])
            A_v_angular = view(A, rows[7:9], nq + (1:nv))
            b_v_angular = view(b, rows[7:9])
            A_v_linear = view(A, rows[10:12], nq + (1:nv))
            b_v_linear = view(b, rows[10:12])
            push!(bodydata,
                BodyData{T}(body,
                    path_to_body,
                    J_geometric,
                    A_q_angular,
                    b_q_angular,
                    A_q_linear,
                    b_q_linear,
                    A_v_angular,
                    b_v_angular,
                    A_v_linear,
                    b_v_linear,
               ))
            row += 12
        end
        new{T}(bodydata, J_q_to_v, A, b, x0, b0)
    end
end

function update_linearization!(data::BodyData, state::MechanismState, J_q_to_v)
    geometric_jacobian!(data.J_geometric, state, data.path_to_body)
    A_mul_B!(data.A_q_angular, angular(data.J_geometric), parent(J_q_to_v))
    A_mul_B!(data.A_q_linear, linear(data.J_geometric), parent(J_q_to_v))
    data.A_v_angular .= angular(data.J_geometric)
    data.A_v_linear .= linear(data.J_geometric)
    H = transform_to_root(state, data.body)
    r = RodriguesVec(rotation(H))
    data.b_q_angular .= (r.sx, r.sy, r.sz)
    data.b_q_linear .= translation(H)
    T = twist_wrt_world(state, data.body)
    data.b_v_angular .= angular(T)
    data.b_v_linear .= linear(T)
    nothing
end

function update_linearization!(bodymap::BodyMap, state::MechanismState)
    RigidBodyDynamics.configuration_derivative_to_velocity_jacobian!(bodymap.J_q_to_v, state)
    for data in bodymap.body_data
        update_linearization!(data, state, bodymap.J_q_to_v)
    end
    copy!(bodymap.x0, configuration(state))
    copy!(bodymap.x0, CartesianRange((num_positions(state)+(1:num_velocities(state)),)), velocity(state), CartesianRange((1:num_velocities(state),)))
    A_mul_B!(bodymap.b0, bodymap.A, bodymap.x0)
    bodymap.b .= bodymap.b .- bodymap.b0;
    nothing
end

_unwrap(x::Real, y::Real) = x + (mod((y - x) + π, 2π) - π)

function _unwrap!(dest::BodyData, src::BodyData)
    dest.b_q_angular .= _unwrap.(src.b_q_angular, dest.b_q_angular)
    dest.b_v_angular .= _unwrap.(src.b_v_angular, dest.b_v_angular)
end

function _unwrap!(dest::BodyMap, src::BodyMap)
    for i in eachindex(dest.body_data)
        _unwrap!(dest.body_data[i], src.body_data[i])
    end
end

mutable struct IKMimicWorkspace{T}
    reference::BodyMap{T}
    result::BodyMap{T}
    H::Matrix{T}
    H_fact::LinAlg.Cholesky{T, Matrix{T}}
    p1::Vector{T}
    f1::Vector{T}
    f2::Vector{T}
    x2::Vector{T}

    function IKMimicWorkspace(reference::BodyMap{T}, result::BodyMap{T}) where {T}
        H = eye(T, size(result.A, 2), size(result.A, 2))
        H_fact = cholfact!(H)
        p1 = zeros(T, size(reference.A, 1))
        f1 = zeros(T, size(result.A, 2))
        f2 = zeros(T, size(result.A, 2))
        x2 = zeros(T, size(result.A, 2))
        new{T}(reference, result, H, H_fact, p1, f1, f2, x2)
    end
end

function IKMimicWorkspace(reference::MechanismState, result::MechanismState, bodynames::AbstractVector{<:AbstractString})
    IKMimicWorkspace(BodyMap(reference, bodynames), BodyMap(result, bodynames))
end

struct SensitivityWorkspace{T}
    A2tA1::Matrix{T}
    J::Matrix{T}

    function SensitivityWorkspace{T}(reference_nx::Integer, result_nx::Integer) where {T}
        new{T}(zeros(T, result_nx, reference_nx),
               zeros(T, result_nx, reference_nx))
    end

end


function mimic_update!(work::IKMimicWorkspace)
    _unwrap!(work.reference, work.result)
    A1 = work.reference.A
    b1 = work.reference.b
    A2 = work.result.A
    b2 = work.result.b

    A_mul_B!(work.p1, A1, work.reference.x0)
    work.p1 .= work.p1 .+ b1

    At_mul_B!(work.H, A2, A2)

    At_mul_B!(work.f1, A2, work.p1)
    At_mul_B!(work.f2, A2, b2)
    work.f1 .= work.f1 .- work.f2
    work.H_fact = cholfact!(Hermitian(work.H))
    A_ldiv_B!(work.x2, work.H_fact, work.f1)

end

function ik_mimic!(result::MechanismState, reference::MechanismState, work::IKMimicWorkspace, iters=1)
    update_linearization!(work.reference, reference)
    for i in 1:iters
        update_linearization!(work.result, result)
        mimic_update!(work)
        copy!(result, work.x2)
    end
    nothing
end

function ik_mimic!(result::MechanismState, reference::MechanismState, bodynames::AbstractVector{<:AbstractString}, iters=1)
    work = IKMimicWorkspace(BodyMap(reference, bodynames),
                            BodyMap(result, bodynames))
    ik_mimic!(result, reference, work, iters)
end

function sensitivity(work::IKMimicWorkspace{T}) where {T}
    sensitivity!(
        SensitivityWorkspace{T}(
            size(work.reference.A, 2), size(work.result.A, 2)),
        work)
end

function sensitivity!(sensitivity_work::SensitivityWorkspace,
                      work::IKMimicWorkspace)
    At_mul_B!(sensitivity_work.A2tA1, work.result.A, work.reference.A)
    A_ldiv_B!(sensitivity_work.J, work.H_fact, sensitivity_work.A2tA1)
    sensitivity_work.J
end


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
