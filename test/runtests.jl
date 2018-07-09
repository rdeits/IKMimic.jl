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
            set_configuration!(state2,
                clamp.(configuration(state2), bounds))
            for i in 1:10
                IKMimic.ik_mimic!(state2, state1, matching_bodies)
            end
            @test Vector(state2) ≈ Vector(state1) rtol=5e-3
        end
    end

    @testset "in-place interface" begin
        srand(1)
        for i in 1:100
            rand!(state1)
            set_configuration!(state1,
                clamp.(configuration(state1), bounds))
            rand!(state2)
            set_configuration!(state2,
                clamp.(configuration(state2), bounds))
            work = IKMimic.IKMimicWorkspace(state1, state2, matching_bodies)
            for i in 1:10
                IKMimic.ik_mimic!(state2, state1, work)
            end
            @test Vector(state2) ≈ Vector(state1) rtol=5e-3
            @test @allocated(IKMimic.ik_mimic!(state2, state1, work)) < 200
        end
    end

    # @testset "
end

