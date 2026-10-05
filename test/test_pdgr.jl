using Test, LinearAlgebra, Serialization

@testset "PDGR integration (no conic solver)" begin
    options = PDGROptions(max_steps=1, fw_max_iteration=50,
        witness_max_length=100000, lmo_nb=1, lmo_max_iter=10, verbose=0)
    param = Param(time_limit=0.0, seed=27, log_level=0, pdgr=options)
    for filename in filter(f -> startswith(f, "state_") && endswith(f, ".jl"),
                           readdir(joinpath(@__DIR__, "../benchmark")))
        data = ExactEntanglement.loadBenchmark(filename)
        result = solvePDGR(data.ρ, data.dims, param)
        @test result.status == "time_limit"
        @test result.config.seed == 27
        @test result.config.max_steps == 1
        @test result.ent_bound == 0 && result.sep_bound == 1
    end

    rho = ComplexF64[1 0 0 1; 0 0 0 0; 0 0 0 0; 1 0 0 1] / 2
    result = solvePDGR(rho, [2, 2], param; time_limit=Inf, seed=0)
    @test result.ent_bound > 0
    @test result.ent_bound <= 2/3 <= result.sep_bound
    @test result.config.seed == 0
    @test result.config.verbose == 0
    @test result.config.fw_max_iteration == options.fw_max_iteration

    @testset "CLI options and result artifacts" begin
        include(joinpath(@__DIR__, "../scripts/run_experiment.jl"))
        old_args = copy(ARGS)
        try
            empty!(ARGS)
            append!(ARGS, ["-s", "state_033.jl", "-a", "PDGR", "-t", "0",
                          "--seed", "9", "--log-level", "0", "--pdgr-mode", "sep",
                          "--pdgr-max-steps", "3", "--pdgr-witness-max-length", "64"])
            args = Base.invokelatest(parseCommandline)
            mktempdir() do dir
                withenv("EXACTENT_RESULTS_DIR" => dir,
                        "MOSEKLM_LICENSE_FILE" => joinpath(dir, "no-license")) do
                    @test runEntangle(args) == 0
                end
                path = joinpath(dir, "state_033.jl_PDGR")
                text = read(path, String)
                @test occursin("algo: PDGR", text)
                @test occursin("lb_relx: NaN", text)
                @test occursin("ub_relx: 1.0", text)
                @test occursin("status: time_limit", text)
                @test occursin("relaxation: none", text)
                stored = deserialize(path * ".jls")
                @test stored.mode == :sep
                @test stored.config.max_steps == 3
                @test stored.config.seed == 9
                @test stored.config.witness_max_length == 64
            end
        finally
            empty!(ARGS)
            append!(ARGS, old_args)
        end
    end
end
