using Test
using ExactEntanglement

@testset "ExactEntanglement" begin
    include("test_mathutils.jl")
    include("test_lift.jl")
    include("test_sbb.jl")
    include("test_params.jl")
    # Needs a Mosek licence; skipped automatically when none is configured.
    include("test_integration.jl")
end
