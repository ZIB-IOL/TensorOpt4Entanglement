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

            @testset "DDPS+ gives a valid lower bound" begin
                @test runEntangle(args("DDPS+")) == 0
                @test 0 < readbound("state_133.jl_DDPS+", "glblb") < 1
            end

            @testset "Alt-SDP gives a valid upper bound" begin
                @test runEntangle(args("Alt-SDP")) == 0
                @test 0 < readbound("state_133.jl_Alt-SDP", "glbub") < 1
            end

            @testset "GHZ bounds respect the analytic threshold" begin
                # the m-party GHZ white-noise threshold is known in closed form
                ghz(m) = 1 - 1 / (1 + 2.0^(m - 1))
                @test ghz(3) ≈ 0.8
                ghzargs = merge(args("DDPS+"), Dict{String,Any}("state" => "state_033.jl"))
                @test runEntangle(ghzargs) == 0
                lb = readbound("state_033.jl_DDPS+", "glblb")
                # DDPS+ is a relaxation, so its lower bound cannot exceed the truth
                @test lb <= ghz(3) + 1e-6
                @test lb > 0.5                      # and it is not vacuous
            end

            @testset "memory is attributed to each level" begin
                @test runEntangle(args("CP")) == 0
                txt = read(joinpath(outdir, "state_133.jl_CP"), String)
                num(k) = parse(Float64, match(Regex("$k: (\\S+)"), txt).captures[1])
                # CP calls the oracle, so :cp must contain :lmo, and :total both
                @test num("mem_lmo_calls") > 0
                @test num("mem_lmo_alloc_gib") <= num("mem_cp_alloc_gib") + 1e-9
                @test num("mem_cp_alloc_gib") <= num("mem_total_alloc_gib") + 1e-9
                @test num("mem_total_peak_rss_mib") > 0
                @test num("mem_lmo_model_nnz") > 0
            end

            @testset "a legacy code writes the canonical filename" begin
                legacy = merge(args("DDPS+"), Dict{String,Any}("algo" => "RLT"))
                @test runEntangle(legacy) == 0
                @test isfile(joinpath(outdir, "state_133.jl_DDPS+"))
                @test !isfile(joinpath(outdir, "state_133.jl_RLT"))
            end

            @testset "the bounds bracket each other" begin
                # DDPS+ lower bound must not exceed the Alt-SDP upper bound
                @test readbound("state_133.jl_DDPS+", "glblb") <=
                      readbound("state_133.jl_Alt-SDP", "glbub") + 1e-6
            end
        finally
            delete!(ENV, "EXACTENT_RESULTS_DIR")
        end
    end
end
