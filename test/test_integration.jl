using Test, JuMP, MosekTools, LinearAlgebra, Random
import Mosek
const E = ExactEntanglement

# The solver-backed algorithms need a Mosek licence. Skip cleanly without one
# so the rest of the suite still runs in CI.
function mosek_available()
    try
        m = Model(Mosek.Optimizer); set_silent(m)
        @variable(m, x >= 1.0); @objective(m, Min, x)
        optimize!(m)
        return termination_status(m) == OPTIMAL
    catch
        return false
    end
end

if !mosek_available()
    @info "Mosek licence not available - skipping integration tests. " *
          "Set MOSEKLM_LICENSE_FILE to enable them."
else
    @testset "integration (Mosek)" begin
        outdir = mktempdir()
        ENV["EXACTENT_RESULTS_DIR"] = outdir
        try
            args(algo) = Dict{String,Any}(
                "state" => "state_133.jl", "algo" => algo, "time-limit" => 60.0,
                "log-level" => 0, "maxnnodes" => 100, "minnnodes" => 1, "maxrounds" => 100,
                "heur-ladmm1-maxiter" => 16, "heur-ladmm-maxiter" => 8,
                "heur-manopt-maxiter" => 150, "loop" => -3)

            readbound(file, key) = parse(Float64,
                match(Regex("$key: (\\S+)"), read(joinpath(outdir, file), String)).captures[1])

            @testset "RLT (DDPS+) gives a valid lower bound" begin
                @test runEntangle(args("RLT")) == 0
                @test 0 < readbound("state_133.jl_RLT", "glblb") < 1
            end

            @testset "Alt-SDP gives a valid upper bound" begin
                @test runEntangle(args("A")) == 0
                @test 0 < readbound("state_133.jl_A", "glbub") < 1
            end

            @testset "the bounds bracket each other" begin
                # DDPS+ lower bound must not exceed the Alt-SDP upper bound
                @test readbound("state_133.jl_RLT", "glblb") <=
                      readbound("state_133.jl_A", "glbub") + 1e-6
            end
        finally
            delete!(ENV, "EXACTENT_RESULTS_DIR")
        end
    end
end
