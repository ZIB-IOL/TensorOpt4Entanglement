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
