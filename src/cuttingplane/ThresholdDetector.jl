

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
    realmaster::Bool
    mastercolumns::Vector{Int}
    columnkeys::Dict{Matrix{Float64},Int}
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
        detector.ispersistent = fill(false, length(purestates))
        detector.poolstats = []
        detector.round = 0
        detector.realmaster = false
        detector.mastercolumns = Int[]
        detector.columnkeys = Dict{Matrix{Float64},Int}()
        return detector
    end
end

function addBatchStates(detector::ThresholdEntanglementDetector, purestates, substates, addtoPool=false)
    append!(detector.purestates, purestates)
    append!(detector.substates, substates)
    append!(detector.ispersistent, fill(false, length(purestates)))

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
        empty!(detector.persistentInds)
    else
        println("clear non persistent states ", length(detector.persistentInds))
        detector.purestates = [detector.purestates[ind] for ind in detector.persistentInds]
        detector.substates = [detector.substates[ind] for ind in detector.persistentInds]
        detector.ispersistent = fill(true, length(detector.persistentInds))
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

"Skip a zero-iteration lift when IR reaches its reserved CP phase."
function irShouldLift(param::Param, singlerun; elapsed = elapsedTime(param))
    return singlerun || param.is_last || elapsed < (1 - param.tratio) * param.time_limit
end

"""
    initialActiveSet(HR, HI, dims, param) -> (detector, weights, nzpure, nzsub)

Build the initial inner approximation `P_1`: the maximally mixed state (retained
across IR iterations) padded with random product states up to
`param.pointsize_bound`, plus the starting convex weights. All `prod(dims)`
computational basis states are retained even when that point bound is smaller.
`nzpure`/`nzsub` are
the tail components carried over when `pointsize_bound > rank_bound`.
"""
function initialActiveSet(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    dimH = reduce(*, dims)
    detector = ThresholdEntanglementDetector(HR, HI, dims, [], [])

    identitystate = Dict(:RE => [Matrix(Diagonal(ones(dim))) / dimH for dim in dims],
                         :IM => [zeros(dim, dim) for dim in dims])
    addRank1State(detector, identitystate, nothing, nothing, true, false)
    npersistent = length(detector.substates)

    purestates_, substates_ = complementStates(dims, param.pointsize_bound, npersistent)
    addBatchStates(detector, purestates_, substates_, false)
    ncomplementpoints = length(detector.substates) - npersistent

    denom = 2 * npersistent + ncomplementpoints
    weights = vcat([2.0 / denom for _ in 1:npersistent], [1.0 / denom for _ in 1:ncomplementpoints])
    @assert length(weights) == length(detector.substates) "Weights and substates must have the same length $(length(weights)) != $(length(detector.substates))"

    selected = sortperm(weights; rev=true)[1:min(length(weights), param.rank_bound)]
    tail = setdiff(eachindex(weights), selected)
    return detector, weights, detector.purestates[tail], detector.substates[tail]
end

"""
    selectTopFactors(weights, rank_bound, arrays...) -> (weights, arrays...)

Keep the `rank_bound` heaviest components and renormalise to a convex
combination, so LADMM is warm-started with factorisation size at most `r`.
"""
function selectTopFactors(weights, rank_bound::Int, arrays...)
    all(a -> length(a) == length(weights), arrays) ||
        throw(DimensionMismatch("Factor arrays and weights must have the same length"))
    isempty(weights) && throw(ArgumentError("LADMM requires a nonempty factorisation"))
    sorted_indices = sortperm(weights, rev = true)
    nfactor = min(length(weights), rank_bound)
    selected = sorted_indices[1:nfactor]
    w = weights[selected]
    w ./= sum(w)
    return (w, map(a -> a[selected], arrays)...)
end

"Trim a real CP packet without splitting its adjacent conjugate pairs."
function selectConjugateFactors(weights, rank_bound::Int, purestates, substates)
    length(weights) == length(purestates) == length(substates) ||
        throw(DimensionMismatch("Factor arrays and weights must have the same length"))
    length(weights) <= rank_bound &&
        return selectTopFactors(weights,rank_bound,substates,purestates)
    singles = Int[]
    pairs = Tuple{Int,Int}[]
    index = 1
    while index <= length(weights)
        P = purestates[index]
        paired = index < length(weights) && weights[index] == weights[index+1] &&
            any(x -> !iszero(imag(x)),P) && size(P) == size(purestates[index+1]) &&
            all(x == conj(y) for (x,y) in zip(P,purestates[index+1]))
        if paired
            push!(pairs,(index,index+1))
            index += 2
        else
            push!(singles,index)
            index += 1
        end
    end
    sort!(singles;by=index -> weights[index],rev=true)
    sort!(pairs;by=pair -> weights[pair[1]]+weights[pair[2]],rev=true)
    single_mass = vcat(0.0,cumsum(weights[singles]))
    pair_mass = vcat(0.0,cumsum([weights[a]+weights[b] for (a,b) in pairs]))
    best_mass, best_singles, best_pairs = -Inf, 0, 0
    # With nonnegative weights, the best packet for each number of pairs
    # uses the heaviest remaining singletons. Enumerating pair counts therefore
    # maximises retained mass under the existing cardinality limit.
    for count in 0:min(length(pairs),fld(rank_bound,2))
        nsingles = min(length(singles),rank_bound-2count)
        mass = single_mass[nsingles+1]+pair_mass[count+1]
        if mass > best_mass
            best_mass, best_singles, best_pairs = mass, nsingles, count
        end
    end
    selected = copy(singles[1:best_singles])
    for (a,b) in pairs[1:best_pairs]
        push!(selected,a,b)
    end
    # At rank one an all-pair packet has no nonempty complete group. Keep
    # the ordinary truncation in that case rather than return an empty lift.
    isempty(selected) && return selectTopFactors(weights,rank_bound,substates,purestates)
    sort!(selected;by=index -> weights[index],rev=true)
    w = weights[selected]
    w ./= sum(w)
    return w, substates[selected], purestates[selected]
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
    addConjugateStates!(detector, addtoPool = false)

Supplement product states with their entrywise conjugates. For real targets,
this lets the master represent the real separable mixture `(P + conj(P))/2`
exactly, instead of compensating imaginary residuals with unrelated columns.
Every added state is a product state on the same subsystems.
"""
function addConjugateStates!(detector, addtoPool = false)
    purestates, substates = [], []
    for i in eachindex(detector.purestates)
        state = detector.purestates[i]
        any(!iszero, imag(state)) || continue
        push!(purestates, conj.(state))
        push!(substates, [conj.(v) for v in detector.substates[i]])
    end
    addBatchStates(detector, purestates, substates, addtoPool)
    return length(purestates)
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

"Construct a consistent lifted warm start after support filtering and rank trimming."
function irWarmStart(purestates, substates, weights, rank_bound, H, ub, multipliers, param;
                     fresh_cp = false, refit_scalar = param.ir_refit_scalar)
    w, sub, pure = fresh_cp && param.cp_real_master && all(iszero,imag(H)) ?
        selectConjugateFactors(weights,rank_bound,purestates,substates) :
        selectTopFactors(weights,rank_bound,substates,purestates)
    z = 1 - ub
    if refit_scalar
        D = size(H, 1)
        mixed = Matrix{ComplexF64}(I, D, D) / D
        B = H - mixed
        Y = sum(w[a] * pure[a] for a in eachindex(w))
        denom = real(dot(B, B))
        denom > eps(Float64) && (z = clamp(real(dot(B, Y - mixed)) / denom, 0.0, 1.0))
    end
    return w, sub, z, multipliers
end

"Continue the lifted iterate when CP returned only its older certificate."
function irNextSeed(snapshot, previous_snapshot, cp_solution, lifted_solution)
    if snapshot === previous_snapshot
        return lifted_solution
    end
    pure, sub, weights, upper = cp_solution
    pure, sub, weights = activeFactors(pure, sub, weights)
    return pure, sub, weights, upper
end

"""
    ladmmLagrangianBound(separateproblem, chi, H, param) -> Float64

Certified lower bound on the white-noise threshold at the LADMM multiplier
`chi` (LADMM sign convention: residual `Psi(x) - A(z) - a`). For the visibility
`z = 1 - noise`, weak duality for `min -z  s.t.  y = (H - I/d) z + I/d,
y separable, z in [0, 1]` gives

    noise >= 1 + min(0, -1 - <chi, H - I/d>) - <chi, I/d> + min_y <chi, y>,

with the last minimum over unit-trace separable `y`. The sBB oracle bounds it
from below as `-U`, where `U >= max_y <-chi, y>` is its certified upper bound,
valid also when the node limit stops the search. Restores the oracle's witness
and cutoff, so the CP master is unaffected. Returns 0.0 when unavailable.
"""
function ladmmLagrangianBound(separateproblem, chi, H, param)
    (isnothing(chi) || remainingTime(param) <= 0) && return 0.0
    cutoff = -norm(chi)
    isfinite(cutoff) || return 0.0
    D = size(H, 1)
    Min = Matrix{ComplexF64}(I, D, D) / D
    Mdir = H - Min
    witness = Dict(:RE => -real(chi), :IM => -imag(chi))
    saved = (separateproblem.H, separateproblem.Hout, separateproblem.cutoffbound)
    local upper
    try
        separateproblem.H = witness
        separateproblem.Hout = witness
        # Every density matrix has Frobenius norm at most one, so this finite
        # cutoff excludes no product state. Infinite cutoffs are not LP data.
        separateproblem.cutoffbound = cutoff
        _, upper, _, _ = separate!(separateproblem, param, 2, false)
    finally
        separateproblem.H, separateproblem.Hout, separateproblem.cutoffbound = saved
    end
    isfinite(upper) || return 0.0
    bound = 1 + min(0.0, -1 - real(dot(chi, Mdir))) - real(dot(chi, Min)) - upper
    return clamp(bound, 0.0, 1.0)
end

"""
    solveIR(HR, HI, dims, param)

Iterative refinement (IR): alternate LADMM on the lifted nonconvex problem with
the cutting-plane master, each warm-starting the other.
"""
function solveIR(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    dimH = prod(dims)
    all(iszero, HI) && HR == Matrix{Float64}(I, dimH, dimH) / dimH &&
        return (0.0, 0.0, 0.0, 0.0, 1.0)
    maxi, singlerun, clearall = refinementBudget(param)

    separateproblem = Problem(HR, HI, dims)
    dimH = separateproblem.dimH
    multipliers = Dict(:RE => zeros(dimH, dimH), :IM => zeros(dimH, dimH))

    ub = 0.5
    seed_upper = ub
    glbub, glblb = 1.0, 0.0
    approxub, approxfeas, approxweights = 1.0, 0.0, 0.0

    remainingTime(param) <= 0 && return glbub, glblb, 1.0, 0.0, 1.0

    detector, weights, nz_purestates, nz_substates = initialActiveSet(HR, HI, dims, param)
    seed_purestates, seed_substates = detector.purestates, detector.substates
    H = HR + im * HI
    snapshot_state = Ref{Union{Nothing,MasterSnapshot}}(nothing)
    ladmm_warm_state = Ref{Any}(nothing)
    priced_columns = (pure=Any[], sub=Any[])

    for i in 1:maxi
        println("Lifting-discretization iteration: $i / $(param.loop)")
        lifted_solution = (seed_purestates, seed_substates, weights, seed_upper)
        if irShouldLift(param, singlerun)
            detector.round = 2 * i - 1
            weights, substates, z, multipliers = irWarmStart(seed_purestates,
                seed_substates, weights, param.rank_bound, H, seed_upper, multipliers, param;
                fresh_cp=param.cp_real_master && all(iszero,HI) &&
                    !isnothing(snapshot_state[]) && isnothing(ladmm_warm_state[]),
                refit_scalar=param.ir_refit_scalar && isnothing(ladmm_warm_state[]))

            candidates = (pure=Any[], sub=Any[])
            candidate_offset = isnothing(snapshot_state[]) ? -Inf : snapshot_state[].offset
            # LADMM on the lifted problem (paper Alg. LADMM)
            purestates, substates, approxub, approxfeas, lifted_weights =
                ladmmSolve(detector, dims, H, substates, weights, z, multipliers, param, i == 1, singlerun;
                    candidate_pool=candidates, candidate_offset=candidate_offset,
                    warm_state=ladmm_warm_state)
            lifted_solution = (purestates, substates, lifted_weights, approxub)
            # A nearly feasible LADMM point already certifies an upper bound by
            # geometric reconstruction, independently of the CP crossover.
            glbub = min(glbub, geometricUpperBound(H, purestates, lifted_weights, approxub, dims))
            if param.ir_ladmm_bound && !isnothing(ladmm_warm_state[])
                glblb = max(glblb, ladmmLagrangianBound(separateproblem,
                    ladmm_warm_state[].multipliers, H, param))
            end

            clearStates(detector, clearall)
            addBatchStates(detector, purestates, substates, param.lazification)
            addBatchStates(detector, candidates.pure, candidates.sub, param.lazification)
            addBatchStates(detector, nz_purestates, nz_substates, false)
        else
            # Keep the current master and heuristic diagnostics. A capped CP
            # pass may have priced one final column without adding it yet.
            addBatchStates(detector, priced_columns.pure, priced_columns.sub, false)
        end

        # Cutting-plane master (paper Alg. CP)
        detector.round = 2 * i
        if param.heur_LADMM_conjugates && !param.cp_real_master && all(iszero, HI)
            addConjugateStates!(detector, param.lazification)
        end
        saved_maxrounds = param.maxrounds
        previous_snapshot = snapshot_state[]
        priced_columns = (pure=Any[], sub=Any[])
        local lb, terminate, weights_sum
        if param.cp_rounds_per_ir > 0 && !singlerun
            param.maxrounds = saved_maxrounds < 0 ? param.cp_rounds_per_ir :
                min(saved_maxrounds, param.cp_rounds_per_ir)
        end
        try
            ub, lb, terminate, purestates, substates, weights, multipliers, weights_sum =
                cuttingPlane(detector, separateproblem, param, 1, singlerun;
                    lower_bound=glblb, snapshot_state=snapshot_state, pricing_pool=priced_columns)
        finally
            param.maxrounds = saved_maxrounds
        end
        glbub = min(glbub, ub)
        glblb = max(glblb, lb)

        # Keep the certified CP support in the next master, while a failed
        # crossover must not erase progress made by the nonconvex solver.
        # A tiny positive LP coefficient can still be essential to the face
        # containing the target. Preserve the complete certificate support in
        # the master; the lifted warm start retains its separate rank filter.
        nz_purestates, nz_substates, _ = activeFactors(purestates, substates, weights; tol=0.0)
        append!(nz_purestates, priced_columns.pure)
        append!(nz_substates, priced_columns.sub)
        seed_purestates, seed_substates, weights, seed_upper = irNextSeed(snapshot_state[],
            previous_snapshot, (purestates, substates, weights, ub), lifted_solution)
        approxweights = weights_sum
        snapshot_state[] === previous_snapshot || (ladmm_warm_state[] = nothing)
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
    dimH = prod(dims)
    all(iszero, HI) && HR == Matrix{Float64}(I, dimH, dimH) / dimH &&
        return (0.0, 0.0, 0.0, 0.0)
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

function dpsOptimizer(param::Param)
    optimizer = Ket.Hypatia.Optimizer{Float64}()
    MOI.set(optimizer, MOI.TimeLimitSec(), param.time_limit < 0 ? Inf : remainingTime(param))
    return optimizer
end

"""
    solveDPS(HR, HI, dims, param)

DPS hierarchy lower bound via `Ket.entanglement_robustness`. The bipartition and
level follow the paper's DPS configuration per subsystem count.
"""
function solveDPS(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    param.time_limit >= 0 && remainingTime(param) <= 0 && return 0.0
    ρ = HR + im * HI
    dimH = reduce(*, dims)
    config = Dict([2, 2, 2] => ([4, 2], 6),
                  [2, 2, 2, 2] => ([4, 4], 2),
                  [2, 2, 2, 2, 2] => ([8, 4], 1))
    haskey(config, dims) || error("no DPS configuration for dims = $dims")
    bipartition, level = config[dims]
    # Ket instantiates this optimizer after constructing the SDP, so model
    # setup consumes the same budget as the solve. Preserve its Hypatia solver.
    val = try
        value, _ = Ket.entanglement_robustness(Matrix(ρ), bipartition, level;
            noise="white", ppt=true, inner=false, verbose=true,
            solver=() -> dpsOptimizer(param))
        value
    catch err
        # Ket reports an unsolved model via Hypatia's raw status. A time-limited
        # primal objective is not a lower certificate; retain the physical zero.
        err isa ErrorException && err.msg == "TimeLimit" && return 0.0
        rethrow()
    end
    # Ket minimises λ subject to ρ + λ·Id lying in the DPS cone, with the
    # UNNORMALISED identity, so that matrix has trace 1 + λ·dimH. Normalising,
    #   (ρ + λ Id)/(1 + λ d) = (1/(1+λd))·ρ + (λd/(1+λd))·(Id/d),
    # and matching (1-z)·ρ + z·(Id/d) gives the mixing parameter
    return dimH * val / (1 + dimH * val)
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

    seed_purestates, seed_substates = detector.purestates, detector.substates

    for i in 1:maxi
        println("Lifting-discretization iteration: $i / $(param.loop)")
        weights, substates, purestates =
            selectTopFactors(weights, param.rank_bound, seed_substates, seed_purestates)

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
        seed_purestates, seed_substates = nz_purestates, nz_substates
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
