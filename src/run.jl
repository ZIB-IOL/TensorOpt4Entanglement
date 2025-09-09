function load_benchmark_data(filename::String)
    benchmark_dir = joinpath(dirname(@__FILE__), "../benchmark")
    filepath = joinpath(benchmark_dir, filename)

    if !isfile(filepath)
        error("File not found: $filepath")
    end

    # Include the file to execute it
    include(filepath)

    # Return the variables defined in the file
    # You may need to adjust these based on what's actually in your files
    return (
        dims = dims,
        ρ = ρ,
        T = T,
        N = N
    )
end

function runEntangle(args)
    println("Loading benchmark data from: $(args["state"])")

    data = load_benchmark_data(args["state"])

    # 创建参数结构体
    param = Param(
        solver = "MSK",
        time_limit = args["time-limit"],
        log_level = args["log-level"],
        maxnnodes = args["maxnnodes"],
        minnnodes = args["minnnodes"],
        maxrounds = args["maxrounds"],
        heur_AD_depth = args["heur-ad-depth"],
        heur_AD_maxiters = args["heur-ad-maxiters"],
        heur_LADMM1_maxiter = args["heur-ladmm1-maxiter"],
        heur_LADMM_maxiter = args["heur-ladmm-maxiter"],
        heur_MANOPT_depth = args["heur-manopt-depth"],
        heur_MANOPT_maxiter = args["heur-manopt-maxiter"],
        loop = args["loop"]
    )

    if param.log_level > 0
        println("Loaded data:")
        println("  T = $(data.T)")
        println("  N = $(data.N)")
        println("  dims = $(data.dims)")
        println("  ρ size = $(size(data.ρ))")
        println("Parameters:")
        println("  algorithm = $(args["algo"])")
        println("  solver = $(param.solver)")
        println("  time_limit = $(param.time_limit)")
        println("  log_level = $(param.log_level)")
        println("  maxrounds = $(param.maxrounds)")
    end

    # 运行纠缠检测
    ρ = Matrix(data.ρ)
    HR = real(ρ)
    HI = imag(ρ)
    dims = data.dims
    nsubs = length(dims)

    dims = collect(data.dims)

    Random.seed!(param.seed)
    # Initialize return variables
    glbub = Inf
    glblb = -Inf
    approxub = Inf
    approxfeas = 0.0
    approxweights = 0
    # adjust parameters based on dimensions
    if  nsubs == 3
        param.time_limit = param.time_limit < 0 ? 3600 : param.time_limit
        param.pointsize_bound = 256 + 10 #min( 100 * (2 * sum(dims) + 1), 2 * prod(dims) + 1)
        param.rank_bound =  256 + 10
        param.maxrounds = 256 + 10
        param.heur_MANOPT_maxiter = 150
        param.heur_MANOPT1_maxiter = 150
        param.heur_LADMM_maxiter = 10
        param.heur_LADMM1_maxiter = 15
        param.maxnnodes = 120
        param.maxeffortnnodes = 70
        param.heur_alternate_iter = 10
        param.heur_alternate1_iter = 15
        param.heur_alternate_maxfail = 1
        param.tratio = 0.1
    elseif nsubs == 4
        param.time_limit = param.time_limit < 0 ? 7200 : param.time_limit
        param.pointsize_bound =  512 + 10  #min( 100 * (2 * sum(dims) + 1), 2 * prod(dims) + 1)
        param.rank_bound = 512 + 10
        param.maxrounds = 512 + 10
        param.heur_MANOPT_maxiter = 150
        param.heur_MANOPT1_maxiter = 200
        param.heur_LADMM_maxiter = 10
        param.heur_LADMM1_maxiter = 15
        param.heur_LADMM_rho = 5
        param.maxnnodes = 100
        param.maxeffortnnodes = 50
        param.heur_alternate_iter = 10
        param.heur_alternate1_iter = 15
        param.heur_alternate_maxfail = 1
        param.tratio = 0.1
    elseif nsubs == 5
        param.time_limit = param.time_limit < 0 ? 10800 : param.time_limit
        param.pointsize_bound = 1096 + 10  #min( 100 * (2 * sum(dims) + 1), 2 * prod(dims) + 1)
        param.rank_bound =  652 + 10
        param.maxrounds = 652 + 10
        param.heur_MANOPT_maxiter = 150
        param.heur_MANOPT1_maxiter = 200
        param.heur_LADMM_maxiter = 4
        param.heur_LADMM1_maxiter = 8
        param.heur_LADMM_rho = 10
        param.maxnnodes = 1
        param.maxeffortnnodes = 15
        param.heur_alternate_iter = 4
        param.heur_alternate1_iter = 8
        param.heur_alternate_maxfail = 1
        param.tratio = 0.15
    elseif nsubs == 6
        param.time_limit = 10800
        param.pointsize_bound =  150 #min( 100 * (2 * sum(dims) + 1), 2 * prod(dims) + 1)
        param.rank_bound =  128
        param.heur_MANOPT_maxiter = 230
        param.heur_LADMM_maxiter = 15
        param.heur_LADMM1_maxiter = 25
        param.maxnnodes = 0
    end
    elapsed_time = @elapsed begin
        if args["algo"] == "LD"
            glbub, glblb, approxub, approxfeas, approxweights =  detectEntanglementThresholdLiftDiscrete(HR, HI, dims, param)
        elseif args["algo"] == "LDual"
            glbub, glblb, approxub, approxfeas, approxweights =  detectEntanglementThresholdLiftDual(HR, HI, dims, param)
        elseif args["algo"] == "LDL"
            param.pool_size = -1
            param.lazification = true
            glbub, glblb, approxub, approxfeas, approxweights =  detectEntanglementThresholdLiftDiscrete(HR, HI, dims, param)
        elseif args["algo"] == "D"
            param.maxrounds = -1
            glbub, glblb, approxub, approxfeas = detectEntanglementThresholdDiscrete(HR, HI, dims, param)
            approxweights = 0  # Reset since this algorithm doesn't return approxweights
        elseif args["algo"] == "A"
            param.heur_alternate_iter = -1
            param.heur_alternate1_iter = -1
            heur_alternate_maxfail = 1
            glbub, glblb, approxub, approxfeas = detectEntanglementThresholdAlternate(HR, HI, dims, param)
            approxweights = 0  # Reset since this algorithm doesn't return approxweights
        elseif args["algo"] == "AD"
            glbub, glblb, approxub, approxfeas = detectEntanglementThresholdHybridSingle(HR, HI, dims, param) # This is called to ensure the hybrid method is also executed
            approxweights = 0  # Reset since this algorithm doesn't return approxweights
        elseif args["algo"] == "PPT"
            glblb = detectEntanglementThresholdPPT(HR, HI, dims, param)
            glbub = 0
            approxub = 0
            approxfeas  = 0
            approxweights = 0
        elseif args["algo"] == "LD0"
            param.loop = -2
            glbub, glblb, approxub, approxfeas, approxweights = detectEntanglementThresholdLiftDiscrete(HR, HI, dims, param)
        elseif args["algo"] == "LD1"
            param.loop = -3
            glbub, glblb, approxub, approxfeas, approxweights = detectEntanglementThresholdLiftDiscrete(HR, HI, dims, param)
        end
    end
    #problem = detectEntanglementThresholdLiftDiscrete(HR, HI, dims, param)
    # Save results to file
    results_dir = joinpath(dirname(@__FILE__), "../results")
    isdir(results_dir) || mkpath(results_dir)
    result_file = joinpath(results_dir, "$(basename(args["state"]))_$(args["algo"])")
    open(result_file, "w") do io
        println(io, "instance: $(args["state"])")
        println(io, "algo: $(args["algo"])")
        println(io, "glbub: $glbub")
        println(io, "approxub: $approxub")
        println(io, "approxfeas: $approxfeas")
        println(io, "glblb: $glblb")
        println(io, "approxweights: $approxweights")
        println(io, "time: $elapsed_time")
    end
    println("Results saved to $result_file")

    println("Processing complete!")

    return 0
end
