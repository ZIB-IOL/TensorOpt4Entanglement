# ---------------------------------------------------------------------------
# Size and memory diagnostics.
#
# Reviewers asked for an estimate of the memory each instance needs. Two
# quantities are recorded with every run:
#
#   * the size of the sBB root relaxation -- variables, constraints and the
#     number of nonzero coefficients in the constraint matrix. This is a
#     property of the model, measured WITHOUT solving, so it is cheap and
#     deterministic (it does not depend on the machine or the time limit).
#   * the peak resident set size of the process, which is what actually has to
#     fit in the cluster's memory limit.
#
# Set EXACTENT_NO_DIAGNOSTICS=1 to skip the relaxation build on very large
# instances; peak RSS is always recorded because it costs nothing.
# ---------------------------------------------------------------------------

"""
    peakRSSMiB()

Peak resident set size of this process, in MiB.
"""
peakRSSMiB() = Sys.maxrss() / 2^20

"""
    functionNNZ(f)

Number of nonzero coefficients contributed by one JuMP constraint function.

Covers real and complex affine expressions (`GenericAffExpr` with any
coefficient type), quadratic expressions, bare variables, constants, and the
vector/matrix containers used by cone constraints. There is deliberately no
generic fallback: an unrecognised type should be a visible MethodError rather
than silently counting zero.
"""
functionNNZ(f::GenericAffExpr) = length(f.terms)
functionNNZ(f::GenericQuadExpr) = length(f.terms) + length(f.aff.terms)
functionNNZ(f::AbstractJuMPScalar) = 1
functionNNZ(f::Number) = 0
functionNNZ(f::AbstractArray) = sum(functionNNZ, f; init = 0)

"""
    modelStats(model) -> (nvars, ncons, nnz)

Count variables, scalar/vector constraints, and nonzero coefficients.
"""
function modelStats(model::Model)
    nvars = num_variables(model)
    ncons = 0
    nnz = 0
    for (F, S) in list_of_constraint_types(model)
        for cref in all_constraints(model, F, S)
            ncons += 1
            nnz += functionNNZ(constraint_object(cref).func)
        end
    end
    return nvars, ncons, nnz
end

"""
    relaxationStats(HR, HI, dims, param) -> NamedTuple

Build the sBB root relaxation for this instance and measure its size without
solving it. Returns variables, constraints, nonzeros and the build time, or
zeros when diagnostics are disabled or the build fails.
"""
function relaxationStats(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    empty = (nvars = 0, ncons = 0, nnz = 0, build_s = 0.0)
    get(ENV, "EXACTENT_NO_DIAGNOSTICS", "0") == "1" && return empty
    try
        t0 = time()
        problem = Problem(HR, HI, dims)
        ss = StateSeparator(problem, param)
        root = createRootNode(problem.dims, problem.BST, problem.Zdims,
                              param.feas_tol, problem.globalZBs, problem.fixvars)
        stateseparatorAddNode!(ss, root)
        optmodel = initRelaxationNode(ss, root, -1.0)
        nvars, ncons, nnz = modelStats(optmodel.model)
        return (nvars = nvars, ncons = ncons, nnz = nnz, build_s = time() - t0)
    catch e
        @warn "relaxation diagnostics failed" exception = (e, catch_backtrace())
        return empty
    end
end

# ---------------------------------------------------------------------------
# Per-phase memory accounting.
#
# Peak RSS alone says how much the process needed, not which algorithmic level
# needed it. Each level is wrapped in `withPhase`, which records for that phase:
#
#   calls        how many times the level was entered
#   alloc        bytes allocated while inside it (churn, not peak)
#   live         live heap at exit, the largest seen
#   rss          process peak RSS at exit, the largest seen
#   model        the largest JuMP model the level built (vars/cons/nnz)
#
# Phases NEST: :cp contains the :lmo calls it makes, and :total contains
# everything. The numbers are therefore inclusive, and :lmo is reported
# separately so its share of :cp is visible.
#
# Peak RSS is a high-water mark, so the growth of `rss` across a phase is what
# that phase added to the process's peak.
# ---------------------------------------------------------------------------

mutable struct PhaseStat
    calls::Int
    alloc::Int
    live::Int
    rss::Float64
    nvars::Int
    ncons::Int
    nnz::Int
    sawnnz::Bool
end
PhaseStat() = PhaseStat(0, 0, 0, 0.0, 0, 0, 0, false)

const PHASES = Dict{Symbol,PhaseStat}()
const PHASE_ORDER = (:total, :cp, :lmo, :ladmm)

"Forget all recorded phase statistics (called once per run)."
function resetPhases!()
    empty!(PHASES)
    return nothing
end

phaseStat(name::Symbol) = get!(PHASES, name, PhaseStat())

"""
    withPhase(f, name)

Run `f()`, attributing its allocations and memory high-water marks to `name`.
Returns exactly what `f` returns, and re-raises unchanged, so wrapping a call
never changes behaviour.
"""
function withPhase(f, name::Symbol)
    st = phaseStat(name)
    st.calls += 1
    before = Base.gc_bytes()
    try
        return f()
    finally
        st.alloc += max(Base.gc_bytes() - before, 0)
        st.live = max(st.live, Base.gc_live_bytes())
        st.rss = max(st.rss, peakRSSMiB())
    end
end

"""
    notePhaseModel!(name, model; nnz = false)

Record the size of a JuMP model built by phase `name`, keeping the largest seen.
Variable and constraint counts are cheap; the nonzero count walks every
constraint, so it is taken only once per phase (the first time `nnz = true` is
passed) to keep the instrumentation off the hot path.
"""
function notePhaseModel!(name::Symbol, model::Model; nnz::Bool = false)
    st = phaseStat(name)
    st.nvars = max(st.nvars, num_variables(model))
    ncons = 0
    for (F, S) in list_of_constraint_types(model)
        ncons += num_constraints(model, F, S)
    end
    st.ncons = max(st.ncons, ncons)
    if nnz && !st.sawnnz
        try
            _, _, n = modelStats(model)
            st.nnz = max(st.nnz, n)
            st.sawnnz = true
        catch e
            @warn "model nnz count failed" exception = e
        end
    end
    return nothing
end

"""
    phaseReport()

Lines of `key: value` describing every recorded phase, for the result file.
"""
function phaseReport()
    lines = String[]
    for name in PHASE_ORDER
        haskey(PHASES, name) || continue
        st = PHASES[name]
        p = "mem_$(name)"
        push!(lines, "$(p)_calls: $(st.calls)")
        push!(lines, "$(p)_alloc_gib: $(round(st.alloc / 2^30, digits = 3))")
        push!(lines, "$(p)_live_mib: $(round(st.live / 2^20, digits = 1))")
        push!(lines, "$(p)_peak_rss_mib: $(round(st.rss, digits = 1))")
        st.nvars > 0 && push!(lines, "$(p)_model_nvars: $(st.nvars)")
        st.ncons > 0 && push!(lines, "$(p)_model_ncons: $(st.ncons)")
        st.nnz > 0 && push!(lines, "$(p)_model_nnz: $(st.nnz)")
    end
    return lines
end
