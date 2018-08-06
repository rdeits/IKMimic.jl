using IKMimic
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end
using RigidBodyDynamics

@testset "simple_atlas -> simple_atlas" begin
    mech1 = parse_urdf(Float64, joinpath(@__DIR__, "urdf", "simple_atlas.urdf"))
    mech2 = parse_urdf(Float64, joinpath(@__DIR__, "urdf", "simple_atlas.urdf"))
    state1 = MechanismState(mech1)
    state2 = MechanismState(mech2)
    matching_bodies =  ["pelvis", "r_foot_sole", "l_foot_sole", "r_hand_mount", "l_hand_mount"]
    bounds = collect(Iterators.flatten(position_bounds(j) for j in joints(mech1)))


    @testset "basic interface" begin
        srand(1)
        for i in 1:10
            rand!(state1)
            set_configuration!(state1,
                clamp.(configuration(state1), bounds))
            rand!(state2)
            # If the pelvis orientation is more than π radians off, there's no reason
            # to expect the mimic to get the right number of 2π offsets
            configuration(state2)[3] = configuration(state1)[3] + (rand() - 0.5) * 2π
            set_configuration!(state2,
                clamp.(configuration(state2), bounds))
            for j in 1:10
                IKMimic.ik_mimic!(state2, state1, matching_bodies)
            end
            @test Vector(state2) ≈ Vector(state1) rtol=5e-3
        end
    end

    @testset "in-place interface" begin
        srand(2)
        for i in 1:100
            rand!(state1)
            set_configuration!(state1,
                clamp.(configuration(state1), bounds))
            rand!(state2)
            # If the pelvis orientation is more than π radians off, there's no reason
            # to expect the mimic to get the right number of 2π offsets
            configuration(state2)[3] = configuration(state1)[3] + (rand() - 0.5) * 2π
            set_configuration!(state2,
                clamp.(configuration(state2), bounds))
            work = IKMimic.IKMimicWorkspace(state1, state2, matching_bodies)
            for j in 1:10
                IKMimic.ik_mimic!(state2, state1, work)
            end
            @test Vector(state2) ≈ Vector(state1) rtol=5e-3
            @test @allocated(IKMimic.ik_mimic!(state2, state1, work)) < 200
        end
    end

    # @testset "sensitivity" begin
    #     srand(1)
    #     for i in 1:10
    #         rand!(state1)
    #         set_configuration!(state1,
    #             clamp.(configuration(state1), bounds))
    #         rand!(state2)
    #         set_configuration!(state2,
    #             clamp.(configuration(state2), bounds))
    #         work = IKMimic.IKMimicWorkspace(state1, state2, matching_bodies)
    #         for j in 1:10
    #             IKMimic.ik_mimic!(state2, state1, work)
    #         end
    #         J = IKMimic.sensitivity(work)
    #         @test J ≈ eye(size(J)...) rtol=1e-3
    #         x1 = copy(Vector(state1))
    #         x2 = copy(Vector(state2))
    #         dx = 1e-5 .* randn(length(x1))
    #         copy!(state1, x2 .+ dx)
    #         IKMimic.ik_mimic!(state2, state1, work)
    #         @test Vector(state2) ≈ x2 .+ J * dx rtol=2e-5
    #     end
    # end
end

