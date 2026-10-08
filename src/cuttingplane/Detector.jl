abstract type AbstractEntanglementDetector end

# Entanglement Detector data

function objective(detector::AbstractEntanglementDetector)
    @objective(detector.model, Max, -detector.b + dot(detector.M[:RE], detector.H[:RE]) + dot(detector.M[:IM], detector.H[:IM]) )
end

function initialLPRelaxation(detector::AbstractEntanglementDetector, param)
    dimH = detector.dimH

    model = Model()
    detector.model = model
    setMosekParam(model, param, true)
    detector.realmaster = param.cp_real_master && all(iszero, detector.H[:IM])
    MI = detector.realmaster ? zeros(AffExpr, dimH, dimH) :
        @variable(model, [1:dimH, 1:dimH] in SkewSymmetricMatrixSpace())
    @variable(model, MR[1:dimH, 1:dimH], Symmetric)
    @variable(model, b)
    detector.M[:RE] = MR
    detector.M[:IM] = MI
    detector.b = b

    normalizationCondition(detector, param)
    # For unit-trace states, M -> M+cI and b -> b+c leave every constraint
    # and the objective unchanged. Remove this free direction from the LP.
    @constraint(model, tr(MR) == 0)
    objective(detector)
    print(length(detector.purestates), " pure states\n")
    empty!(detector.cuts)
    empty!(detector.mastercolumns)
    empty!(detector.columnkeys)
    for (i, state) in enumerate(detector.purestates)
        addCons(detector, state, i)
    end
end

function addCons(detector::AbstractEntanglementDetector, state, index = length(detector.purestates))
    if detector.realmaster
        key = real(state)
        haskey(detector.columnkeys, key) && return false
        detector.columnkeys[key] = index
    end
    cons = @constraint(detector.model, dot(detector.M[:RE], real(state)) + dot(detector.M[:IM], imag(state)) - detector.b <= 0)
    push!(detector.cuts, cons)
    push!(detector.mastercolumns, index)
    return true
end

function poolAdd(detector::AbstractEntanglementDetector, param::Param)
    # LADMM columns can already be active when their pool round becomes
    # eligible. Add only genuinely missing master columns, including when
    # several pool entries describe the same column.
    active = Set((detector.realmaster ? real(state) : state) for state in detector.purestates)
    for (ind, state) in enumerate(detector.poolpurestates)
        if detector.poolstats[ind] > 0 && detector.poolstats[ind] < detector.round
            key = detector.realmaster ? real(state) : state
            key in active && continue
            push!(detector.substates, detector.poolsubstates[ind])
            push!(detector.purestates, detector.poolpurestates[ind])
            push!(detector.ispersistent, false)
            addCons(detector, detector.poolpurestates[ind])
            push!(active,key)
        end
    end
end

function addRank1PrincipleState(detector::AbstractEntanglementDetector, Xvals, Mout, valueb, recordindx = false, addtoPool = false)
    nsubs = detector.nsubs
    sumdim = sum(detector.dims)
    substates = []
    for i in 1:nsubs
        M = Xvals[:RE][i] + im * Xvals[:IM][i]
        eigvals, eigenvecs = eigen(M)
        # Extract the eigenvector corresponding to the maximum eigenvalue
        maxndex = argmax(real(eigvals))   # Index of the maximum eigenvalue
        x = eigenvecs[:, maxndex]     # Corresponding eigenvector
        # normalize the eigenvector
        x = x / norm(x)
        push!(substates, x)
    end
    pstate = [substate * substate' for substate in substates]
    state = foldl(kron, pstate)
    if isnothing(Mout) && isnothing(valueb)
        push!(detector.substates, substates)
        push!(detector.purestates, state)
        push!(detector.ispersistent, recordindx)
        if recordindx
            push!(detector.persistentInds, length(detector.purestates))
        end
        if addtoPool
            push!(detector.poolpurestates, state)
            push!(detector.poolsubstates, substates)
            push!(detector.poolstats, detector.round)
        end
    elseif dot(Mout[:RE], real(state)) + dot(Mout[:IM], imag(state)) > valueb
        push!(detector.substates, substates)
        push!(detector.purestates, state)
        push!(detector.ispersistent, recordindx)
        recordindx && push!(detector.persistentInds, length(detector.purestates))
        if addtoPool
            push!(detector.poolpurestates, state)
            push!(detector.poolsubstates, substates)
            push!(detector.poolstats, detector.round)
        end
        addCons(detector, state)
    end
    return 1
end

function addRank1State(detector::AbstractEntanglementDetector, Xvals, Mout, valueb, recordindx = false, addtoPool = false, pointsize_bound = -1)
    nsubs = detector.nsubs
    sumdim = sum(detector.dims)
    eigenvecs = [[] for i in 1:nsubs]
    for i in 1:nsubs
        M = Xvals[:RE][i] + im * Xvals[:IM][i]
        vals, vecs = eigen(M)
        if !( size(M, 1) == detector.dims[i] && size(M, 2) == detector.dims[i] )
            print("size not matched,", (size(M), detector.dims[i]))
        end
        @assert size(M, 1) == detector.dims[i] && size(M, 2) == detector.dims[i]
        for j in 1:length(vals)
            if abs(vals[j]) < 1e-6
                continue
            end
            @assert( length(vecs[:, j]) == detector.dims[i])
            push!(eigenvecs[i], vecs[:, j])
        end
    end
    ct = 0
    for indices in Iterators.product((1:length(eigenvecs[i]) for i in 1:nsubs)...)
        substates = [eigenvecs[i][indices[i]] for i in 1:nsubs]
        @assert( sumdim == sum( [length(substate) for substate in substates]) )
        for i in 1:nsubs
            @assert( length(substates[i]) ==  detector.dims[i] )
        end
        pstate = [substate * substate' for substate in substates]
        state = foldl(kron, pstate)
        tracestate = tr(state)
        state /= tracestate
        if isnothing(Mout) && isnothing(valueb)
            ct += 1
            push!(detector.substates, substates)
            push!(detector.purestates, state)
            push!(detector.ispersistent, recordindx)
            recordindx && push!(detector.persistentInds, length(detector.purestates))
            if addtoPool
                push!(detector.poolpurestates, state)
                push!(detector.poolsubstates, substates)
                push!(detector.poolstats, detector.round)
            end
        elseif dot(Mout[:RE], real(state)) + dot(Mout[:IM], imag(state)) > valueb
            ct += 1
            push!(detector.substates, substates)
            push!(detector.purestates, state)
            push!(detector.ispersistent, recordindx)
            recordindx && push!(detector.persistentInds, length(detector.purestates))
            if addtoPool
                push!(detector.poolpurestates, state)
                push!(detector.poolsubstates, substates)
                push!(detector.poolstats, detector.round)
            end
            addCons(detector, state)
        end
        if pointsize_bound > 0 && ct >= pointsize_bound
            break
        end
    end
    return ct
end

function checkTerminationGap(detector::AbstractEntanglementDetector, primalobj, cutactiveness, cutdualactiveness, valueb, dualobj, param;
                             master_upper = primalobj)
    addstate = cutactiveness > valueb
    terminate = false
    newdualobj = primalobj + min(0.0, valueb - cutdualactiveness)
    if master_upper - max(dualobj,newdualobj) <= param.master_obj_tol
        print("the certified cutting-plane gap is closed\n")
        terminate = true
    end
    return terminate, addstate, newdualobj
end
