"""
    cuttingPlane(detector, separateproblem, param, effortlevel = 0, singlerun = false)

Cutting-plane algorithm (paper Alg. CP).

Maintains a finite inner approximation `P` of the extreme rays of the separable
tensor cone. Each round solves the master relaxation over `P` for an upper bound
`ub_relx`, calls the sBB linear-minimisation oracle ([`separate!`](@ref)) for a
violated extreme point, and adds it to `P`. The oracle's own lower bound gives
the Lagrangian bound `lb_relx = ub_relx + b_lower`.

Returns
`(ub_relx, lb_relx, terminate, purestates, substates, weights, Mout, weights_sum)`
where `weights` are the normalised master duals (`λ_p`), `Mout` the dual matrix
`y*`, and `weights_sum` their pre-normalisation total.

With `singlerun = true` the routine stops after the first master solve; this is
the one-shot crossover used by `-a LD1`.

`lower_bound` carries a lower bound already obtained by preceding CP passes in
IR. It does not trigger an additional relaxation solve.
"""
cuttingPlane(detector::AbstractEntanglementDetector, separateproblem, param::Param,
             effortlevel = 0, singlerun = false; lower_bound = 0.0, snapshot_state = nothing) =
    withPhase(:cp) do
        cuttingPlane_(detector, separateproblem, param, effortlevel, singlerun;
            lower_bound=lower_bound, snapshot_state=snapshot_state)
    end

"A matched master solution, independent of later changes to the column pool."
struct MasterSnapshot
    upper::Float64
    objective::Float64
    offset::Float64
    purestates
    substates
    weights::Vector{Float64}
    witness::Dict{Symbol,Matrix{Float64}}
    weights_sum::Float64
    residual::Float64
end

function mixedMasterSnapshot(detector)
    pure, sub = [], []
    for inds in Iterators.product((1:d for d in detector.dims)...)
        factors = [ComplexF64[j == inds[k] for j in 1:d] for (k, d) in enumerate(detector.dims)]
        push!(sub, factors)
        push!(pure, foldl(kron, [v * v' for v in factors]))
    end
    D = detector.dimH
    zero_witness = Dict(:RE => zeros(D, D), :IM => zeros(D, D))
    return MasterSnapshot(1.0, 0.0, 0.0, pure, sub, fill(1.0 / D, D), zero_witness, 1.0, 0.0)
end

function masterSnapshot(detector, upper, param)
    has_values(detector.model) && has_duals(detector.model) && isfinite(upper) || return nothing
    raw = Float64[abs(dual(cut)) for cut in detector.cuts]
    total = sum(raw)
    isfinite(total) && total > 0 || return nothing
    pure, sub, weights = [], [], Float64[]
    for (a, index) in enumerate(detector.mastercolumns)
        P, factors, w = detector.purestates[index], detector.substates[index], raw[a] / total
        if detector.realmaster && any(!iszero, imag(P))
            push!(pure, P, conj.(P))
            push!(sub, factors, [conj.(v) for v in factors])
            push!(weights, w / 2, w / 2)
        else
            push!(pure, P); push!(sub, factors); push!(weights, w)
        end
    end
    H = detector.H[:RE] + im * detector.H[:IM]
    D = detector.dimH
    mixed = Matrix{ComplexF64}(I, D, D) / D
    Y = sum(weights[a] * pure[a] for a in eachindex(weights))
    residual = norm(Y - ((1 - upper) * H + upper * mixed))
    residual <= 10 * param.feas_tol || return nothing
    M = Dict(:RE => Matrix(value.(detector.M[:RE])), :IM => Matrix(value.(detector.M[:IM])))
    b = value(detector.b)
    scale = dot(M[:RE], real(H - mixed)) + dot(M[:IM], imag(H))
    isfinite(scale) && scale > 0 && abs(scale - 1) <= 10 * param.feas_tol || return nothing
    M[:RE] ./= scale; M[:IM] ./= scale; b /= scale
    objective = dot(M[:RE], detector.H[:RE]) + dot(M[:IM], detector.H[:IM]) - b
    return MasterSnapshot(upper, objective, b, copy(pure), copy(sub), weights, M, total, residual)
end

function snapshotResult(snapshot, lower, terminate)
    # IR changes multiplier signs in place; protect the saved CP convention.
    witness = Dict(part => copy(values) for (part,values) in snapshot.witness)
    return snapshot.upper, lower, terminate, snapshot.purestates,
        snapshot.substates, snapshot.weights, witness, snapshot.weights_sum
end

"""
    stableMasterWitness(detector, snapshot, param)

Choose a small-norm normalised witness within `100*master_obj_tol` of the LP
objective. The master can have many nearly optimal witnesses with very
different norms; reducing their norm improves the conditioning of the next
LADMM call and of pricing. Keep the LP's upper bound and decomposition: the
quadratic problem's duals are not convex weights. The oracle uses the selected
witness's actual objective when calculating its Lagrangian lower bound.
"""
function stableMasterWitness(detector, snapshot, param)
    remainingTime(param) > 5.0 || return snapshot
    model = detector.model
    lpobjective = objective_function(model)
    floor = @constraint(model, lpobjective >= snapshot.objective - 100param.master_obj_tol)
    @objective(model, Min, sum(x^2 for x in detector.M[:RE]) + sum(x^2 for x in detector.M[:IM]))
    try
        status, _, _, _ = solveMSK(model, param, true)
        status in (RelaxOptimal, RelaxFeasible) || return snapshot
        M = Dict(:RE => Matrix(value.(detector.M[:RE])), :IM => Matrix(value.(detector.M[:IM])))
        mixed = Matrix{Float64}(I, detector.dimH, detector.dimH) / detector.dimH
        scale = dot(M[:RE], detector.H[:RE] - mixed) + dot(M[:IM], detector.H[:IM])
        isfinite(scale) && scale > 0 && abs(scale - 1) <= 10param.feas_tol || return snapshot
        M[:RE] ./= scale
        M[:IM] ./= scale
        b = value(detector.b) / scale
        objective = dot(M[:RE], detector.H[:RE]) + dot(M[:IM], detector.H[:IM]) - b
        isfinite(objective) || return snapshot
        return MasterSnapshot(snapshot.upper, objective, b, snapshot.purestates, snapshot.substates,
            snapshot.weights, M, snapshot.weights_sum, snapshot.residual)
    finally
        delete(model, floor)
        @objective(model, Max, lpobjective)
    end
end

function cuttingPlane_(detector::AbstractEntanglementDetector, separateproblem, param::Param,
                       effortlevel = 0, singlerun = false;
                       lower_bound = 0.0, snapshot_state = nothing)
    snapshot = isnothing(snapshot_state) || isnothing(snapshot_state[]) ?
        mixedMasterSnapshot(detector) : snapshot_state[]
    lower, terminate = max(0.0, lower_bound), false
    remainingTime(param) <= 0 && return snapshotResult(snapshot, lower, terminate)
    initialLPRelaxation(detector, param)
    trace = cpTraceSink()
    iter = 0
    has_current = false
    stable_snapshot = nothing
    try
        while remainingTime(param) > 0
            if isTimeLimitNearlyReached(param)
                param.lazification && poolAdd(detector, param)
                extendTimeLimit!(param)
            end
            notePhaseModel!(:cp, detector.model; nnz=true)
            status, _, objective, upper = solveMSK(detector.model, param, false)
            status in (RelaxOptimal, RelaxFeasible) || break
            current = masterSnapshot(detector, upper, param)
            isnothing(current) && break
            # Use the preceding pass only as a fallback. A valid new pass must
            # supply its own warm start, even if the global upper bound is
            # better; otherwise IR restarts from the same factors indefinitely.
            if !has_current || current.upper <= snapshot.upper
                snapshot = current
                has_current = true
            end
            isnothing(snapshot_state) || (snapshot_state[] = snapshot)
            snapshot.upper - lower <= param.master_obj_tol && (terminate = true; break)
            earlyStopping(detector, current.upper, param) && (terminate = true; break)
            singlerun && break
            remainingTime(param) <= 0 && break

            certify = !param.is_last && param.cp_certify_every > 0 && iter % param.cp_certify_every == 0
            if certify
                current = stableMasterWitness(detector, current, param)
                stable_snapshot = current
            end
            if snapshot.upper == current.upper
                snapshot = current
                isnothing(snapshot_state) || (snapshot_state[] = snapshot)
            end
            separateproblem.H = current.witness
            separateproblem.Hout = current.witness
            separateproblem.cutoffbound = current.offset
            level = param.is_last ? 2 : (certify ? 1 : 0)
            saved_nodes = param.maxnnodes
            certify && (param.maxnnodes = 1)
            local activeness, oracle_upper, state, Xvals
            try
                activeness, oracle_upper, state, Xvals = separate!(separateproblem, param, level, false)
            finally
                param.maxnnodes = saved_nodes
            end
            terminate, addstate, candidate_lower = checkTerminationGap(detector,
                current.objective, activeness, oracle_upper, current.offset, lower, param)
            lower = max(lower, candidate_lower)
            traceRow!(trace, iter, param.is_last, snapshot.upper, lower,
                candidate_lower - current.upper, length(detector.purestates))
            iter += 1
            if terminate || remainingTime(param) <= 0 ||
               (!param.is_last && param.maxrounds >= 0 && iter >= min(param.maxrounds, 2 * detector.dimH^2 + 1))
                break
            end
            addstate && !isnothing(Xvals) || break
            addRank1PrincipleState(detector, Xvals, current.witness,
                current.offset, false, param.lazification)
        end
    finally
        traceClose!(trace)
        if param.lazification
            empty!(detector.poolpurestates)
            empty!(detector.poolsubstates)
            empty!(detector.poolstats)
        end
    end
    if has_current && !isnothing(snapshot_state) && !singlerun && !terminate &&
       !param.is_last && param.cp_certify_every > 0 &&
       snapshot !== stable_snapshot
        snapshot = stableMasterWitness(detector, snapshot, param)
        snapshot_state[] = snapshot
    end
    return snapshotResult(snapshot, lower, terminate)
end
