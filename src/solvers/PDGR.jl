"Options for the PDGR implementation in lib/PDGR."
const PDGROptions = PDGR.Options

"""
    solvePDGR(rho, dims, param=Param(); options=param.pdgr, kwargs...)

Compute PDGR white-noise bounds. Shared `Param` settings supply `time_limit`,
`seed`, and `log_level`; PDGR-specific settings live in `param.pdgr`.
Explicit keyword arguments take precedence over both. Returns the full PDGR
result, including witnesses and history. No conic solver is invoked.
"""
function solvePDGR(rho::Union{AbstractMatrix,AbstractVector}, dims,
                   param::Param=Param(); options::PDGROptions=param.pdgr, kwargs...)
    shared = (time_limit = param.time_limit < 0 ? Inf : param.time_limit,
              seed = param.seed, verbose = param.log_level)
    return PDGR.solve(rho, dims; options, merge(shared, (; kwargs...))...)
end

solvePDGR(HR::AbstractMatrix, HI::AbstractMatrix, dims, param::Param; kwargs...) =
    solvePDGR(complex.(HR, HI), dims, param; kwargs...)

# CLI PDGR settings have their own prefix so sBB/IR iteration limits retain
# their existing meaning. Unspecified settings use the supplied both.jl preset.
function pdgrOptions(args, base::PDGROptions=PDGROptions())
    fields = (:bound_atol, :max_steps, :fw_epsilon, :fw_max_iteration,
              :lmo_nb, :lmo_max_iter, :witness_max_length)
    settings = Dict{Symbol,Any}(k => getfield(base, k) for k in fieldnames(PDGROptions))
    for field in fields
        key = "pdgr-" * replace(string(field), "_" => "-")
        value = get(args, key, nothing)
        value === nothing || (settings[field] = value)
    end
    return PDGROptions(; settings...)
end

function runPDGRBenchmark(args, data, param)
    options = pdgrOptions(args, param.pdgr)
    mode = Symbol(get(args, "pdgr-mode", "both"))
    elapsed = @elapsed result = solvePDGR(data.ρ, data.dims, param; options, mode)
    results_dir = get(ENV, "EXACTENT_RESULTS_DIR", joinpath(@__DIR__, "../../results"))
    mkpath(results_dir)
    path = joinpath(results_dir, "$(basename(args["state"]))_PDGR")
    # Preserve the full result separately; the standard text result remains
    # readable by the existing table and figure scripts.
    Serialization.serialize(path * ".jls", result)
    open(path, "w") do io
        println(io, "instance: $(args["state"])")
        println(io, "algo: PDGR")
        println(io, "ub_relx: $(something(result.sep_bound, NaN))")
        println(io, "lb_relx: $(something(result.ent_bound, NaN))")
        println(io, "ub_heur: NaN\nfeas_heur: NaN\nweights_sum: NaN")
        println(io, "time: $elapsed")
        println(io, "status: $(result.status)")
        println(io, "gap: $(something(result.gap, NaN))")
        println(io, "mode: $(result.mode)")
        println(io, "structure: $(result.structure)")
        println(io, "steps: $(length(result.history))")
        println(io, "relaxation: none")
        println(io, "seed: $(result.config.seed)")
        println(io, "julia: $(VERSION)")
        println(io, "host: $(gethostname())")
        println(io, "pdgr_source: https://github.com/ZIB-IOL/EntanglementDetection.jl")
        println(io, "frankwolfe_version: $(pkgversion(PDGR.EntanglementDetection.FrankWolfe))")
        println(io, "ket_version: $(pkgversion(Ket))")
        println(io, "peak_rss_mib: $(round(peakRSSMiB(), digits=1))")
        for field in fieldnames(PDGROptions)
            println(io, "pdgr_$field: $(getfield(result.config, field))")
        end
    end
    println("Results saved to $path (certificates and history: $path.jls)")
    return 0
end
