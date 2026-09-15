# ---------------------------------------------------------------------------
# Command-line driver: load a benchmark state, tune parameters for its size,
# run the requested algorithm, write the result file.
# ---------------------------------------------------------------------------

"""
    loadBenchmark(filename) -> (; dims, ρ, T, N)

Evaluate a benchmark instance from `benchmark/`.

The file is evaluated into a throwaway module rather than `include`d into this
one: `include` inside a function creates the bindings in a newer world age than
the calling frame, so reading them back directly fails on Julia >= 1.12 with
"the binding may be too new".
"""
function loadBenchmark(filename::String)
    filepath = joinpath(dirname(@__FILE__), "../benchmark", filename)
    isfile(filepath) || error("File not found: $filepath")

    # Benchmark files declare `using Ket` themselves but were written to be
    # included into this module, so they also rely on LinearAlgebra/Random
    # being ambiently available (e.g. `norm`). Reproduce that environment.
    sandbox = Module(:BenchmarkInstance)
    Base.eval(sandbox, :(using LinearAlgebra, Random))
    Base.include(sandbox, filepath)
    get(sym) = Base.invokelatest(getfield, sandbox, sym)

    return (dims = get(:dims), ρ = get(:ρ), T = get(:T), N = get(:N))
end

"""
Per-subsystem-count parameter presets, following the experimental settings in
the paper (time limits 1/2/3 h and factorisation size r = 266/522/662 for
m = 3/4/5; sBB node limits 120/100/1 and gap-closing limits 70/50/15).

`time_limit` is applied only when the caller passed a negative time limit.
"""
const SIZE_PRESETS = Dict(
    3 => (time_limit = 3600.0,  pointsize_bound = 266,  rank_bound = 266,
          heur_MANOPT_maxiter = 150, heur_MANOPT1_maxiter = 150,
          heur_LADMM_maxiter = 10, heur_LADMM1_maxiter = 15,
          maxnnodes = 120, maxeffortnnodes = 70,
          heur_alternate_iter = 10, heur_alternate1_iter = 15,
          heur_alternate_maxfail = 1, tratio = 0.1),
    4 => (time_limit = 7200.0,  pointsize_bound = 522,  rank_bound = 522,
          heur_MANOPT_maxiter = 150, heur_MANOPT1_maxiter = 200,
          heur_LADMM_maxiter = 10, heur_LADMM1_maxiter = 15, heur_LADMM_rho = 5.0,
          maxnnodes = 100, maxeffortnnodes = 50,
          heur_alternate_iter = 10, heur_alternate1_iter = 15,
          heur_alternate_maxfail = 1, tratio = 0.1),
    5 => (time_limit = 10800.0, pointsize_bound = 1106, rank_bound = 662,
          heur_MANOPT_maxiter = 150, heur_MANOPT1_maxiter = 200,
          heur_LADMM_maxiter = 4, heur_LADMM1_maxiter = 8, heur_LADMM_rho = 10.0,
          maxnnodes = 1, maxeffortnnodes = 15,
          heur_alternate_iter = 4, heur_alternate1_iter = 8,
          heur_alternate_maxfail = 1, tratio = 0.15),
    6 => (pointsize_bound = 150, rank_bound = 128,
          heur_MANOPT_maxiter = 230,
          heur_LADMM_maxiter = 15, heur_LADMM1_maxiter = 25,
          maxnnodes = 0),
)

"""
Factorisation size r selected by the `LADMM_<r>` codes (the m = 5 rank sweep of
the paper's low-rank table).
"""
const RANK_SWEEP = Dict("LADMM_$r" => r for r in (400, 500, 600, 700, 800, 900))

"""
    applySizePreset!(param, nsubs, algo)

Apply the preset for `nsubs` subsystems in place. `maxrounds` tracks the
factorisation size (the CP iteration limit inside IR is set to r).
"""
function applySizePreset!(param::Param, nsubs::Int, algo::String)
    algo = resolveAlgorithm(algo)   # so a legacy code still selects its r
    preset = get(SIZE_PRESETS, nsubs, nothing)
    isnothing(preset) && return param

    for (field, value) in pairs(preset)
        field === :time_limit && continue
        setfield!(param, field, value)
    end
    if haskey(preset, :time_limit)
        # m == 6 overrode the time limit unconditionally; 3-5 only as a default.
        param.time_limit = (nsubs == 6 || param.time_limit < 0) ? preset.time_limit : param.time_limit
    elseif nsubs == 6
        param.time_limit = 10800.0
    end

    # r, and with it the CP iteration limit. The LDR* rank sweep is an m == 5
    # experiment only; at other sizes the preset's own r stands.
    if nsubs != 6
        r = nsubs == 5 ? get(RANK_SWEEP, algo, param.rank_bound) : param.rank_bound
        param.rank_bound = r
        param.maxrounds = r
    end
    return param
end

"""
Algorithm codes accepted by `-a`. These are a stable external contract:
`jobs.sh`, the filenames under `results/`, and `scripts/make_tables.py` all key
off them, so add codes rather than renaming existing ones.

| code          | what it runs                                          |
|---------------|-------------------------------------------------------|
| `Alt-SDP`     | alternating SDP                                       |
| `LADMM`       | one refinement iteration = LADMM + one CP crossover    |
| `LADMM_400`…`LADMM_900` | LADMM at factorisation size r (m = 5 sweep) |
| `CP`          | standalone cutting plane                              |
| `IR`          | iterative refinement (LADMM + CP, with lazification)  |
| `DPS`         | DPS hierarchy lower bound via Ket.jl                  |
| `DDPS+`       | tensor-RLT lower bound at the sBB root                |
| `DDPS`        | as `DDPS+` with the McCormick families removed        |
| `CP-DDPS`, `IR-DDPS` | `CP` / `IR` with a DDPS-only oracle            |
| `IR-nolazy`, `IR-clear` | IR variants not named in the paper          |
| `Alt-SDP+CP`  | alternating SDP inside the refinement loop            |
| `DualALM`     | experimental dual ALM, not in the paper               |

The codes are the paper's algorithm names. The shorthand used before they were
aligned (`A`, `LD1`, `D`, `LDL`, `PPT`, `RLT`, `LDR0`…`LDR5`, …) is still
accepted on the command line and when reading result files, so the runs
published with the paper still load; see `LEGACY_ALIASES`.

Each entry maps `(HR, HI, dims, param)` to
`(glbub, glblb, approxub, approxfeas, approxweights)` in the paper's notation
`(ub_relx, lb_relx, ub_heur, feas_heur, Σλ)`.
"""
const ALGORITHMS = Dict{String,Function}(
    # --- the algorithms compared in the paper ------------------------------
    "Alt-SDP" => function (HR, HI, dims, p)
        p.heur_alternate_iter = -1
        p.heur_alternate1_iter = -1
        p.heur_alternate_maxfail = 1
        ub, lb, aub, afeas = solveAltSDP(HR, HI, dims, p)
        return ub, lb, aub, afeas, 0
    end,
    "LADMM"   => function (HR, HI, dims, p)
        p.loop = -3                       # one IR iteration = LADMM + crossover
        solveIR(HR, HI, dims, p)
    end,
    "CP"      => function (HR, HI, dims, p)
        p.maxrounds = -1
        ub, lb, aub, afeas = solveCP(HR, HI, dims, p)
        return ub, lb, aub, afeas, 0
    end,
    "IR"      => function (HR, HI, dims, p)
        p.pool_size = -1
        p.lazification = true
        solveIR(HR, HI, dims, p)
    end,
    "DPS"     => (HR, HI, dims, p) -> (0, solveDPS(HR, HI, dims, p), 0, 0, 0),
    "DDPS+"   => (HR, HI, dims, p) -> (0, solveDDPSPlus(HR, HI, dims, p), 0, 0, 0),

    # --- variants not given a name in the paper ----------------------------
    "IR-nolazy"  => (HR, HI, dims, p) -> solveIR(HR, HI, dims, p),
    "IR-clear"   => function (HR, HI, dims, p)
        p.loop = -2                       # one IR iteration, active set cleared
        solveIR(HR, HI, dims, p)
    end,
    "Alt-SDP+CP" => (HR, HI, dims, p) -> solveAltSDPCP(HR, HI, dims, p),
    "DualALM"    => (HR, HI, dims, p) -> solveDualALM(HR, HI, dims, p),
)

"""
Codes used before the algorithm names were aligned with the paper. Accepted on
the command line and when reading result files, so the runs published with the
paper -- whose filenames carry the old codes -- still resolve.
"""
const LEGACY_ALIASES = Dict(
    "A" => "Alt-SDP", "LD1" => "LADMM", "D" => "CP", "LDL" => "IR",
    "PPT" => "DPS", "RLT" => "DDPS+",
    "LD" => "IR-nolazy", "LD0" => "IR-clear", "AD" => "Alt-SDP+CP",
    "LDual" => "DualALM",
    "RLT_DDPS" => "DDPS", "D_DDPS" => "CP-DDPS", "LDL_DDPS" => "IR-DDPS",
    ("LDR$(i-1)" => "LADMM_$r" for (i, r) in enumerate((400, 500, 600, 700, 800, 900)))...,
)

"""
    resolveAlgorithm(code) -> String

Canonical name for `code`, accepting the legacy shorthand.
"""
resolveAlgorithm(code::AbstractString) = get(LEGACY_ALIASES, code, String(code))


"""
    withRelaxation(code, mode)

The same algorithm with the sBB oracle restricted to the plain DDPS
relaxation. Running both gives the DDPS vs DDPS+ comparison directly.
"""
function withRelaxation(code::String, mode::Symbol)
    base = ALGORITHMS[code]
    return function (HR, HI, dims, p)
        p.relaxation = mode
        base(HR, HI, dims, p)
    end
end

# The rank sweep is LADMM at a fixed factorisation size; the size itself is
# applied by applySizePreset! from RANK_SWEEP.
for code in keys(RANK_SWEEP)
    ALGORITHMS[code] = ALGORITHMS["LADMM"]
end

# DDPS-only counterparts, for the ablation against DDPS+. The paper calls the
# unstrengthened outer approximation simply DDPS.
ALGORITHMS["DDPS"]    = withRelaxation("DDPS+", :ddps)
ALGORITHMS["CP-DDPS"] = withRelaxation("CP", :ddps)
ALGORITHMS["IR-DDPS"] = withRelaxation("IR", :ddps)

function runEntangle(args)
    println("Loading benchmark data from: $(args["state"])")
    data = loadBenchmark(args["state"])
    algo = resolveAlgorithm(args["algo"])

    param = Param(
        solver = "MSK",
        time_limit = args["time-limit"],
        log_level = args["log-level"],
        maxnnodes = args["maxnnodes"],
        minnnodes = args["minnnodes"],
        maxrounds = args["maxrounds"],
        heur_LADMM1_maxiter = args["heur-ladmm1-maxiter"],
        heur_LADMM_maxiter = args["heur-ladmm-maxiter"],
        heur_MANOPT_maxiter = args["heur-manopt-maxiter"],
        loop = args["loop"],
    )

    ρ = Matrix(data.ρ)
    HR, HI = real(ρ), imag(ρ)
    dims = collect(data.dims)

    if param.log_level > 0
        println("Loaded data:")
        println("  T = $(data.T)")
        println("  N = $(data.N)")
        println("  dims = $(data.dims)")
        println("  ρ size = $(size(data.ρ))")
        println("Parameters:")
        println("  algorithm = $algo")
        println("  solver = $(param.solver)")
        println("  time_limit = $(param.time_limit)")
        println("  log_level = $(param.log_level)")
        println("  maxrounds = $(param.maxrounds)")
    end

    param.relaxation = Symbol(get(args, "relaxation", "ddpsplus"))
    param.relaxation in (:ddps, :ddpsplus) ||
        error("--relaxation must be ddps or ddpsplus, got $(param.relaxation)")
    param.seed = get(args, "seed", param.seed)

    Random.seed!(param.seed)
    applySizePreset!(param, length(dims), algo)

    haskey(ALGORITHMS, algo) || error("unknown algorithm $algo; expected one of $(sort(collect(keys(ALGORITHMS))))")

    # Size of the sBB root relaxation, measured without solving it.
    stats = relaxationStats(HR, HI, dims, param)
    if param.log_level > 0
        println("Root relaxation: $(stats.nvars) vars, $(stats.ncons) constraints, ",
                "$(stats.nnz) nonzeros, $(round(stats.cbf_bytes/2^20, digits=2)) MiB as CBF ",
                "(measured in $(round(stats.build_s, digits=2)) s)")
    end
    glbub, glblb, approxub, approxfeas, approxweights = Inf, -Inf, Inf, 0.0, 0
    resetPhases!()
    elapsed_time = @elapsed begin
        glbub, glblb, approxub, approxfeas, approxweights =
            withPhase(:total) do
                ALGORITHMS[algo](HR, HI, dims, param)
            end
    end

    # Overridable so tests and CI do not write into the tracked results/ tree.
    results_dir = get(ENV, "EXACTENT_RESULTS_DIR", joinpath(dirname(@__FILE__), "../results"))
    isdir(results_dir) || mkpath(results_dir)
    result_file = joinpath(results_dir, "$(basename(args["state"]))_$(algo)")
    open(result_file, "w") do io
        println(io, "instance: $(args["state"])")
        println(io, "algo: $algo")
        # Field names are the paper's symbols. Runs published before the
        # rename used glbub / glblb / approxub / approxfeas / approxweights;
        # the reader in scripts/tables/common.py accepts either.
        println(io, "ub_relx: $glbub")
        println(io, "ub_heur: $approxub")
        println(io, "feas_heur: $approxfeas")
        println(io, "lb_relx: $glblb")
        println(io, "weights_sum: $approxweights")
        println(io, "time: $elapsed_time")
        # provenance: what was run, and with which settings
        println(io, "relaxation: $(param.relaxation)")
        println(io, "seed: $(param.seed)")
        println(io, "julia: $(VERSION)")
        println(io, "host: $(gethostname())")
        # size and memory (see Diagnostics.jl)
        println(io, "relax_nvars: $(stats.nvars)")
        println(io, "relax_ncons: $(stats.ncons)")
        println(io, "relax_nnz: $(stats.nnz)")
        println(io, "relax_cbf_bytes: $(stats.cbf_bytes)")
        println(io, "peak_rss_mib: $(round(peakRSSMiB(), digits=1))")
        # per-level memory: :total contains :cp, which contains :lmo (see Diagnostics.jl)
        for line in phaseReport()
            println(io, line)
        end
    end
    println("Results saved to $result_file")
    println("Processing complete!")
    return 0
end
