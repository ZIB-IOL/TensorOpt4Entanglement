using Test
using ExactEntanglement

@testset "ExactEntanglement" begin
    include("test_mathutils.jl")
    include("test_lift.jl")
    include("test_sbb.jl")
    include("test_benchmarks.jl")
    include("test_params.jl")
    include("test_diagnostics.jl")
    include("test_pdgr_core.jl")
    include("test_pdgr.jl")
    # Needs a Mosek licence; skipped automatically when none is configured.
    include("test_integration.jl")
end
