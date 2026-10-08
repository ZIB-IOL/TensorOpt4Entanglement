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
             effortlevel = 0, singlerun = false; lower_bound = 0.0, snapshot_state = nothing,
             pricing_pool = nothing) =
    withPhase(:cp) do
        cuttingPlane_(detector, separateproblem, param, effortlevel, singlerun;
            lower_bound=lower_bound, snapshot_state=snapshot_state, pricing_pool=pricing_pool)
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

"Select a sparse matched LP certificate from the solver's available results."
function masterSnapshot(detector, upper, param; result = nothing)
    model = detector.model
    if isnothing(result)
        # MOSEK can return both its interior-point and basic LP solutions.
        # Older MosekTools versions expose the dense interior-point solution
        # first. Selecting an already computed basic solution avoids discarding
        # essential tiny coefficients or growing the next IR master needlessly.
        candidates = MasterSnapshot[]
        for index in 1:result_count(model)
            snapshot = masterSnapshot(detector, upper, param; result=index)
            isnothing(snapshot) || push!(candidates,snapshot)
        end
        isempty(candidates) && return nothing
        best_upper = minimum(snapshot.upper for snapshot in candidates)
        eligible = filter(snapshot -> snapshot.upper <= best_upper + param.master_obj_tol, candidates)
        return eligible[argmin([(length(snapshot.weights),snapshot.upper) for snapshot in eligible])]
    end
    feasible = (MOI.FEASIBLE_POINT,MOI.NEARLY_FEASIBLE_POINT)
    primal_status(model;result=result) in feasible &&
        dual_status(model;result=result) in feasible || return nothing
    upper = dual_objective_value(model;result=result)
    primal = objective_value(model;result=result)
    isfinite(upper) && isfinite(primal) && primal <= upper + param.obj_tol || return nothing
    raw = Float64[abs(dual(cut;result=result)) for cut in detector.cuts]
    total = sum(raw)
    isfinite(total) && total > 0 || return nothing
    pure, sub, weights = [], [], Float64[]
    for (a, index) in enumerate(detector.mastercolumns)
        P, factors, w = detector.purestates[index], detector.substates[index], raw[a] / total
        iszero(w) && continue
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
    M = Dict(:RE => Matrix(value.(detector.M[:RE];result=result)),
             :IM => Matrix(value.(detector.M[:IM];result=result)))
    b = value(detector.b;result=result)
    isfinite(b) && all(isfinite,M[:RE]) && all(isfinite,M[:IM]) || return nothing
    scale = dot(M[:RE], real(H - mixed)) + dot(M[:IM], imag(H))
    isfinite(scale) && scale > 0 && abs(scale - 1) <= 10 * param.feas_tol || return nothing
    M[:RE] ./= scale; M[:IM] ./= scale; b /= scale
    objective = dot(M[:RE], detector.H[:RE]) + dot(M[:IM], detector.H[:IM]) - b
    isfinite(objective) || return nothing
    if upper < 0
        # An interior separable target can have a negative unrestricted LP
        # value. Interpolate its decomposition back to H rather than return
        # an unphysical negative white-noise threshold.
        alpha = 1 / (1 - upper)
        weights .*= alpha
        if alpha < 1
            basis = mixedMasterSnapshot(detector)
            append!(pure, basis.purestates)
            append!(sub, basis.substates)
            append!(weights, (1 - alpha) .* basis.weights)
        end
        residual *= alpha
        upper = 0.0
    end
    return MasterSnapshot(upper, objective, b, copy(pure), copy(sub), weights, M, total, residual)
end

"Retain a violated final pricing column for the next IR master."
function retainPricingColumn!(pool, detector, Xvals, witness, offset)
    isnothing(pool) && return
    factors = [begin
        matrix = Hermitian(Xvals[:RE][k] + im * Xvals[:IM][k])
        values, vectors = eigen(matrix)
        x = vectors[:, argmax(values)]
        x / norm(x)
    end for k in eachindex(detector.dims)]
    P = foldl(kron, [x * x' for x in factors])
    dot(witness[:RE], real(P)) + dot(witness[:IM], imag(P)) > offset || return
    push!(pool.pure, P)
    push!(pool.sub, factors)
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
                       lower_bound = 0.0, snapshot_state = nothing, pricing_pool = nothing)
    validateBoundTolerances(param)
    snapshot = isnothing(snapshot_state) || isnothing(snapshot_state[]) ?
        mixedMasterSnapshot(detector) : snapshot_state[]
    lower, terminate = max(0.0, lower_bound), false
    remainingTime(param) <= 0 && return snapshotResult(snapshot, lower, terminate)
    initialLPRelaxation(detector, param)
    trace = cpTraceSink()
    iter = 0
    has_current = false
    stable_snapshot = nothing
    stabilized_final = false
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

            periodic = param.cp_certify_every > 0 && iter % param.cp_certify_every == 0
            certify = !param.is_last && periodic
            unstabilized = current
            if periodic || (param.is_last && param.cp_certify_every > 0 && !stabilized_final)
                current = stableMasterWitness(detector, current, param)
                stable_snapshot = current
                stabilized_final |= param.is_last
            end
            if snapshot.upper == current.upper
                snapshot = current
                isnothing(snapshot_state) || (snapshot_state[] = snapshot)
            end
            level = param.is_last ? 2 : (certify ? 1 : 0)
            local activeness, oracle_upper, state, Xvals
            local addstate, candidate_lower
            for attempt in 1:2
                separateproblem.H = current.witness
                separateproblem.Hout = current.witness
                separateproblem.cutoffbound = current.offset
                saved_nodes = param.maxnnodes
                certify && (param.maxnnodes = 1)
                try
                    activeness, oracle_upper, state, Xvals = separate!(separateproblem, param, level, false)
                finally
                    param.maxnnodes = saved_nodes
                end
                terminate, addstate, candidate_lower = checkTerminationGap(detector,
                    current.objective, activeness, oracle_upper, current.offset, lower, param;
                    master_upper=snapshot.upper)
                lower = max(lower, candidate_lower)
                # A near-optimal regularised witness can have no violated
                # product while its objective loss still exceeds the desired
                # gap. Price the original LP witness before stopping.
                if attempt == 1 && !terminate && !addstate && current !== unstabilized && remainingTime(param) > 0
                    current = unstabilized
                else
                    break
                end
            end
            traceRow!(trace, iter, param.is_last, snapshot.upper, lower,
                candidate_lower - current.upper, length(detector.purestates))
            iter += 1
            addstate && !isnothing(Xvals) &&
                retainPricingColumn!(pricing_pool, detector, Xvals, current.witness, current.offset)
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
            # Inactive columns are eligible in later IR rounds. Preserve the
            # pool across passes, with duplicate columns and capacity managed
            # independently of the current master and its certificate support.
            compactPool!(detector, param)
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
