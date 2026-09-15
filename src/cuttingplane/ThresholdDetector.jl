

mutable struct ThresholdEntanglementDetector <: AbstractEntanglementDetector
    H
    dims::Vector{Int64}
    dimH::Int
    nsubs::Int
    M
    b
    model
    purestates
    substates
    cuts
    persistentInds
    ispersistent
    poolpurestates
    poolsubstates
    poolstats
    round
    function ThresholdEntanglementDetector(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, purestates, substates)
        dimH = reduce(*, dims)
        nsubs = length(dims)
        H = Dict(:RE=>HR, :IM=>HI)
        M = Dict(:RE => zeros(AffExpr, 0, 0), :IM => zeros(AffExpr, 0, 0))
        b = AffExpr()
        detector = new(H, dims, dimH, nsubs, M, b)
        detector.purestates = copy(purestates)
        detector.substates = copy(substates)
        detector.cuts = []
        detector.persistentInds = []
        detector.poolpurestates = []
        detector.poolsubstates = []
        detector.ispersistent = []
        detector.poolstats = []
        detector.round = 0
        return detector
    end
end

function addBatchStates(detector::ThresholdEntanglementDetector, purestates, substates, addtoPool=false)
    append!(detector.purestates, purestates)
    append!(detector.substates, substates)
    append!(detector.ispersistent, [false] * length(purestates))

    if addtoPool
        append!(detector.poolpurestates, purestates)
        append!(detector.poolsubstates, substates)
        append!(detector.poolstats, fill(detector.round, length(purestates)))
    end
end

function clearStates(detector::ThresholdEntanglementDetector, clearall = false)
    if clearall
        println("clearall states")
        detector.purestates = []
        detector.substates = []
        detector.ispersistent = []
    else
        println("clear non persistent states ", length(detector.persistentInds))
        detector.purestates = [detector.purestates[ind] for ind in detector.persistentInds]
        detector.substates = [detector.substates[ind] for ind in detector.persistentInds]
        detector.ispersistent  = [true] * length(detector.persistentInds)
        detector.persistentInds = [ind for ind in 1:length(detector.persistentInds)]
    end
    detector.cuts = []
end

function normalizationCondition(detector::ThresholdEntanglementDetector, param)
    # normalization condition
    @constraint(detector.model, dot(detector.M[:RE], detector.H[:RE] - Diagonal(ones(detector.dimH) / detector.dimH)) + dot(detector.M[:IM], detector.H[:IM]) == 1 )
end

function earlyStopping(detector::ThresholdEntanglementDetector, primalobj, param)
    isEarlyStopping = primalobj < param.master_obj_tol
    if isEarlyStopping
        println("early stopping: the state is not entangled: $(primalobj)\n")
    end
    return isEarlyStopping
end

function updateProblem(detector::ThresholdEntanglementDetector, problem, primalobj, param)
    t = primalobj + param.master_obj_tol
    proximal = Dict(:RE => (1 - t) * detector.H[:RE] + t * Diagonal(ones(detector.dimH) / detector.dimH), :IM=> (1 - t) * detector.H[:IM])
    problem.proximal = proximal
end

function complementStates(dims, maxpoints, npoints)
    dimH = reduce(*, dims)
    pointbound = min(dimH * dimH * 2 + 1, maxpoints)
    ncomplement = pointbound - npoints
    if ncomplement <= 0
        return [], []
    end
    substates = []
    purestates = []
    for i in 1:ncomplement
        xs = []
        prod = 1.0
        for dim in dims
            x = randn(ComplexF64, dim)
            x ./= norm(x)
            push!(xs, x)
            prod = kron(prod, x)
        end
        prod = prod * prod'
        prod = prod / tr(prod)
        push!(substates, xs)
        push!(purestates, prod)
    end
    return purestates, substates
end

# ---------------------------------------------------------------------------
# Drivers for the white-noise mixing threshold problem.
#
# Paper -> code:
#   IR       (iterative refinement)        -> solveIR   [-a LD/LD1/LDL/LDR*]
#   CP       (cutting plane, standalone)   -> solveCP       [-a D]
#   Alt-SDP  (alternating SDP)             -> solveAltSDP      [-a A]
#   Alt-SDP + CP                           -> solveAltSDPCP   [-a AD]
#   DDPS+    (tensor RLT lower bound)      -> solveDDPSPlus            [-a RLT]
#   DPS      (via Ket.jl)                  -> solveDPS            [-a PPT]
#   dual ALM (experimental, not in paper)  -> solveDualALM       [-a LDual]
#
# Bounds returned, in paper notation:
#   glbub     = ub_relx    (upper bound from the convex master relaxation)
#   glblb     = lb_relx    (Lagrangian lower bound, ub_relx + LMO lower bound)
#   approxub  = ub_heur    (objective of the nonconvex lifted solution)
#   approxfeas= feas_heur  (its residual ||A(z) + a - Psi(x)||, 0 => ub_heur valid)
# ---------------------------------------------------------------------------

"""
    refinementBudget(param) -> (maxiter, singlerun, clearall)

Decode `param.loop` into the IR loop budget. The negative sentinels select the
run modes exposed on the command line:

  `-1` unlimited IR iterations (`-a LD`, `-a LDL`)
  `-2` one iteration, clearing the active set  (`-a LD0`)
  `-3` one iteration, keeping the active set   (`-a LD1`, `-a LDR*`; standalone LADMM)
"""
function refinementBudget(param::Param)
    param.loop == -1 && return (10000000, false, false)
    param.loop == -2 && return (1, true, true)
    param.loop == -3 && return (1, true, false)
    return (param.loop, false, false)
end

"""
    initialActiveSet(HR, HI, dims, param) -> (detector, weights, nzpure, nzsub)

Build the initial inner approximation `P_1`: the maximally mixed state (retained
across IR iterations) padded with random product states up to
`param.pointsize_bound`, plus the starting convex weights. `nzpure`/`nzsub` are
the tail components carried over when `pointsize_bound > rank_bound`.
"""
function initialActiveSet(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    dimH = reduce(*, dims)
    detector = ThresholdEntanglementDetector(HR, HI, dims, [], [])

    identitystate = Dict(:RE => [Matrix(Diagonal(ones(dim))) / dimH for dim in dims],
                         :IM => [zeros(dim, dim) for dim in dims])
    addRank1State(detector, identitystate, nothing, nothing, true, false, param.pointsize_bound)
    npersistent = length(detector.substates)

    purestates_, substates_ = complementStates(dims, param.pointsize_bound, npersistent)
    addBatchStates(detector, purestates_, substates_, false)
    ncomplementpoints = length(detector.substates) - npersistent

    denom = 2 * npersistent + ncomplementpoints
    weights = vcat([2.0 / denom for _ in 1:npersistent], [1.0 / denom for _ in 1:ncomplementpoints])
    @assert length(weights) == length(detector.substates) "Weights and substates must have the same length $(length(weights)) != $(length(detector.substates))"

    nnz = max(param.pointsize_bound - param.rank_bound, 0)
    return detector, weights, detector.purestates[end-nnz:end], detector.substates[end-nnz:end]
end

"""
    selectTopFactors(weights, rank_bound, arrays...) -> (weights, arrays...)

Keep the `rank_bound` heaviest components and renormalise to a convex
combination, so LADMM is warm-started with factorisation size at most `r`.
"""
function selectTopFactors(weights, rank_bound::Int, arrays...)
    sorted_weights = sort(weights, rev = true)
    sorted_indices = sortperm(weights, rev = true)
    nfactor = min(length(sorted_weights), rank_bound)
    w = sorted_weights[1:nfactor]
    w ./= sum(w)
    return (w, map(a -> a[sorted_indices][1:nfactor], arrays)...)
end

"""
    activeFactors(purestates, substates, weights; tol = 1e-7)

Support of the master LP's basic optimal solution, i.e. `P_{k+1} = {p : λ_p > 0}`.
"""
function activeFactors(purestates, substates, weights; tol = 1e-7)
    keep = [i for i in 1:length(weights) if weights[i] > tol]
    return purestates[keep], substates[keep], weights[keep]
end

"""
    flipMultipliers!(multipliers)

The LMO and the lifted solver use opposite sign conventions for the dual
matrix; flip on the way back into LADMM.
"""
function flipMultipliers!(multipliers)
    multipliers[:RE] = -multipliers[:RE]
    multipliers[:IM] = -multipliers[:IM]
    return multipliers
end

"""
    solveIR(HR, HI, dims, param)

Iterative refinement (IR): alternate LADMM on the lifted nonconvex problem with
the cutting-plane master, each warm-starting the other.
"""
function solveIR(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    maxi, singlerun, clearall = refinementBudget(param)

    separateproblem = Problem(HR, HI, dims)
    dimH = separateproblem.dimH
    multipliers = Dict(:RE => zeros(dimH, dimH), :IM => zeros(dimH, dimH))

    ub = 0.5
    glbub, glblb = Inf, -Inf
    approxub, approxfeas, approxweights = 1.0, 0.0, 0.0

    detector, weights, nz_purestates, nz_substates = initialActiveSet(HR, HI, dims, param)

    for i in 1:maxi
        println("Lifting-discretization iteration: $i / $(param.loop)")
        detector.round = 2 * i - 1
        weights, substates = selectTopFactors(weights, param.rank_bound, detector.substates)

        # LADMM on the lifted problem (paper Alg. LADMM)
        purestates, substates, approxub, approxfeas =
            ladmmSolve(detector, dims, HR + im * HI, substates, weights, 1 - ub, multipliers, param, i == 1, singlerun)

        clearStates(detector, clearall)
        addBatchStates(detector, purestates, substates, param.lazification)
        addBatchStates(detector, nz_purestates, nz_substates, false)

        # Cutting-plane master (paper Alg. CP)
        detector.round = 2 * i
        ub, lb, terminate, purestates, substates, weights, multipliers, weights_sum =
            cuttingPlane(detector, separateproblem, param, 1, singlerun)
        glbub = min(glbub, ub)
        glblb = max(glblb, lb)

        nz_purestates, nz_substates, weights = activeFactors(purestates, substates, weights)
        approxweights = weights_sum
        flipMultipliers!(multipliers)

        println("Terminated at iteration $i with number of substates: $(length(weights)), glbub: $glbub, glblb: $glblb, approxub: $approxub, approxweights: $approxweights")
        if terminate
            println("Terminated at iteration $i with ub: $ub, lb: $lb")
            break
        end
        if isTimeLimitExceeded(param)
            println("Time limit exceeded, exiting...")
            break
        end
    end
    println("Loop finished $(param.loop)")
    return glbub, glblb, approxub, approxfeas, approxweights
end

"""
    solveCP(HR, HI, dims, param)

Standalone cutting-plane algorithm (CP): no lifted heuristic, one master solve
refined by the sBB oracle until the bounds meet.
"""
function solveCP(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    separateproblem = Problem(HR, HI, dims)
    detector, _, _, _ = initialActiveSet(HR, HI, dims, param)
    ub, lb, _, _, _, _, _, _ = cuttingPlane(detector, separateproblem, param, 1)
    return ub, lb, ub, 0.0
end

"""
    solveAltSDP(HR, HI, dims, param)

SDP-based alternating optimisation (Alt-SDP): one subsystem density matrix per
rank-one component is left free and updated by an SDP. Upper bound only.
"""
function solveAltSDP(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    detector, _, _, _ = initialActiveSet(HR, HI, dims, param)
    weights = ones(length(detector.substates)) / length(detector.substates)
    ub, _, _ = altSDPSolve(dims, HR + im * HI, detector.purestates, detector.substates, weights, param)
    return ub, -Inf64, ub, 0.0
end

"""
    solveDPS(HR, HI, dims, param)

DPS hierarchy lower bound via `Ket.entanglement_robustness`. The bipartition and
level follow the paper's DPS configuration per subsystem count.
"""
function solveDPS(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    ρ = HR + im * HI
    dimH = reduce(*, dims)
    config = Dict([2, 2, 2] => ([4, 2], 6),
                  [2, 2, 2, 2] => ([4, 4], 2),
                  [2, 2, 2, 2, 2] => ([8, 4], 1))
    haskey(config, dims) || error("no DPS configuration for dims = $dims")
    bipartition, level = config[dims]
    val, _ = Ket.entanglement_robustness(Matrix(ρ), bipartition, level;
                                         noise = "white", ppt = true, inner = false, verbose = true)
    return val / dimH
end

"""
    solveDDPSPlus(HR, HI, dims, param)

DDPS+ lower bound: solve the tensor-RLT relaxation once at the sBB root.
"""
function solveDDPSPlus(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    return threshold!(Problem(HR, HI, dims), param, 2)
end

"""
    solveAltSDPCP(HR, HI, dims, param)

Alt-SDP in place of LADMM inside the refinement loop: the alternating SDP
supplies the candidate extreme points that the cutting-plane master then prices.
"""
function solveAltSDPCP(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    singlerun, clearall, maxi = false, false, 10000000

    separateproblem = Problem(HR, HI, dims)
    dimH = separateproblem.dimH
    multipliers = Dict(:RE => zeros(dimH, dimH), :IM => zeros(dimH, dimH))

    glbub, glblb = Inf, -Inf
    approxub, approxweights = 1.0, 0.0

    detector, weights, nz_purestates, nz_substates = initialActiveSet(HR, HI, dims, param)

    for i in 1:maxi
        println("Lifting-discretization iteration: $i / $(param.loop)")
        weights, substates, purestates =
            selectTopFactors(weights, param.rank_bound, detector.substates, detector.purestates)

        ub, purestates, substates = altSDPSolve(dims, HR + im * HI, purestates, substates, weights, param, i == 1)
        glbub = min(glbub, ub)

        clearStates(detector, clearall)
        addBatchStates(detector, purestates, substates, param.lazification)
        addBatchStates(detector, nz_purestates, nz_substates)

        ub, lb, terminate, purestates, substates, weights, multipliers, weights_sum =
            cuttingPlane(detector, separateproblem, param, 1, singlerun)
        glbub = min(glbub, ub)
        glblb = max(glblb, lb)

        nz_purestates, nz_substates, weights = activeFactors(purestates, substates, weights)
        approxweights = weights_sum
        flipMultipliers!(multipliers)

        println("Terminated at iteration $i with number of substates: $(length(weights)), glbub: $glbub, glblb: $glblb, approxub: $approxub, approxweights: $approxweights")
        if terminate
            println("Terminated at iteration $i with ub: $ub, lb: $lb")
            break
        end
        if isTimeLimitExceeded(param)
            println("Time limit exceeded, exiting...")
            break
        end
    end
    return glbub, glblb, glbub, 0.0, approxweights
end

"""
    solveDualALM(HR, HI, dims, param)

Experimental dual augmented-Lagrangian variant (`-a LDual`). Not part of the
paper; kept for comparison. Runs a single dual ALM solve and reports its bound.
"""
function solveDualALM(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    _, singlerun, _ = refinementBudget(param)
    multipliers = 1.0

    detector, weights, _, _ = initialActiveSet(HR, HI, dims, param)
    weights, substates = selectTopFactors(weights, param.rank_bound, detector.substates)

    _, _, approxlb, approxfeas =
        dualALMSolve(detector, dims, HR + im * HI, substates, weights, multipliers, param, true, singlerun)

    return 0, 0, approxlb, approxfeas, 0.0
end
