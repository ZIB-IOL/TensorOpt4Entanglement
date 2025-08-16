


abstract type AbstractEntanglementDetector end

# Entanglement Detector data
#=
mutable struct BaseEntanglementDetector <: AbstractEntanglementDetector H
    dims::Vector{Int64}
    dimH::Int
    nsubs::Int
    M
    b
    model
    purestates
    substates

    function BaseEntanglementDetector(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param, purestates)
        dimH = reduce(*, dims)
        nsubs = length(dims)
        H = Dict(:RE=>HR, :IM=>HI)
        M = Dict(:RE=>  zeros(AffExpr, 0, 0), :IM=>  zeros(AffExpr, 0, 0))
        b =  AffExpr()
        substates = []
        entanglementdetector = new(H, dims, dimH, nsubs, M, b, [], substates)
        entanglementdetector.purestates = copy(purestates)
        entanglementdetector.substates = []
        return entanglementdetector
    end
end
=#


#=
function FWHeuristicActive(stateseparator::StateSeparator, H, Xvals)
    max_iteration = 5
    epsilon = 1e-7
    Hbar = constructFullSol(stateseparator.problem, Xvals)
    res=ED.separable_distance(Hermitian(H), Tuple(stateseparator.problem.dims); verbose = stateseparator.param.log_level > 3, ini_sigma=Hbar, max_iteration, epsilon)
    primal = sqrt(res.primal)
    if primal < stateseparator.primalbd
        if stateseparator.param.log_level > 0
            print("find a better primal bound $(primal) by FW\n")
        end
        stateseparator.primalbd = primal
    end
end
=#


function normalizationCondition(detector::AbstractEntanglementDetector, param)
end

function objective(detector::AbstractEntanglementDetector)
    @objective(detector.model, Max, -detector.b + dot(detector.M[:RE], detector.H[:RE]) + dot(detector.M[:IM], detector.H[:IM]) )
end

function updateProblem(detector::AbstractEntanglementDetector, problem, primalobj, param)
end

function initialLPRelaxation(detector::AbstractEntanglementDetector, param)
    dimH = detector.dimH

    model = Model()
    detector.model = model
    setMosekParam(model, param, true)
    @variable(model, MI[1:dimH, 1:dimH] in SkewSymmetricMatrixSpace())
    @variable(model, MR[1:dimH, 1:dimH], Symmetric)
    @variable(model, b)
    detector.M[:RE] = MR
    detector.M[:IM] = MI
    detector.b = b

    normalizationCondition(detector, param)
    objective(detector)
    print(length(detector.purestates), " pure states\n")
    #cons = @constraint(model, dot(detector.M[:RE], Matrix(I, dimH, dimH) / dimH)  - detector.b <= 0)
    #push!(detector.cuts, cons)
    for state in detector.purestates
        cons = @constraint(model, dot(detector.M[:RE], real(state)) + dot(detector.M[:IM], imag(state)) - detector.b <= 0)
        push!(detector.cuts, cons)
    end
end


function addState(detector::AbstractEntanglementDetector, state)
    @constraint(detector.model, dot(detector.M[:RE], real(state)) + dot(detector.M[:IM], imag(state)) - detector.b <= 0)
    push!(detector.purestates, Hermitian(state))
end


function addCons(detector::AbstractEntanglementDetector, state)
    cons = @constraint(detector.model, dot(detector.M[:RE], real(state)) + dot(detector.M[:IM], imag(state)) - detector.b <= 0)
    push!(detector.cuts, cons)
end

function updatePoolStats(detector::AbstractEntanglementDetector, param::Param)
    for (ind, state) in enumerate(detector.poolpurestates)
        violation = value(dot(detector.M[:RE], real(state)) + dot(detector.M[:IM], imag(state)) - detector.b)
        detector.poolstats[ind] = 0.5 * detector.poolstats[ind] + violation
    end
end

function poolAdd(detector::AbstractEntanglementDetector, param::Param)
    for (ind, state) in enumerate(detector.poolpurestates)
        if detector.poolstats[ind] > 0 && detector.poolstats[ind] < detector.round
            addCons(detector, detector.poolpurestates[ind])
            push!(detector.substates, detector.poolsubstates[ind])
            push!(detector.purestates, detector.poolpurestates[ind])
            push!(detector.ispersistent, false)
        end
    end
end

function searchFilterCons(detector::AbstractEntanglementDetector, poolsize, param::Param)
    bestviolation = 1e-6
    beststateind = -1
    for (ind, state) in enumerate(detector.poolpurestates)
        violation = value(dot(detector.M[:RE], real(state)) + dot(detector.M[:IM], imag(state)) - detector.b)
        if violation >= bestviolation
            beststateind = ind
            bestviolation = violation
        end
    end
    addstate = false
    if beststateind != -1
        addstate = true
        addCons(detector, detector.poolpurestates[beststateind])
        push!(detector.substates, detector.poolsubstates[beststateind])
        push!(detector.purestates, detector.poolpurestates[beststateind])
        push!(detector.ispersistent, false)
    end
    if length(detector.poolpurestates) >= param.pool_size
        detector.poolpurestates = detector.poolpurestates[1:param.pool_size]
        detector.poolsubstates = detector.poolsubstates[1:param.pool_size]
    end
    return addstate, bestviolation
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
            push!(detector.presistentinds, length(detector.purestates))
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
        if addtoPool
            push!(detector.poolpurestates, state)
            push!(detector.poolsubstates, substates)
            push!(detector.poolstats, detector.round)
        end
        addCons(detector, state)
    end
    #check(detector.substates, detector.dims)
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
            push!(detector.presistentinds, length(detector.purestates))
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
            push!(detector.presistentinds, length(detector.purestates))
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
    #check(detector.substates, detector.dims)
    return ct
end



function earlyStopping(detector::AbstractEntanglementDetector, primalobj, param)
    return false
end

function checkTerminationGap(detector::AbstractEntanglementDetector, primalobj, cutactiveness, cutdualactiveness, valueb, dualobj, param)
    addstate = cutactiveness > valueb
    terminate = false
    newdualobj = primalobj + valueb - cutdualactiveness
    if cutdualactiveness < valueb
        print("the state is entangled, because the primal obj is positive and optimal (no violated constraint)\n")
        terminate = true
    else
        if newdualobj > param.master_obj_tol
            #terminate = true
            #print("the state is entangled, because the dual obj is positive\n")
        end
    end
    return terminate, addstate, newdualobj
end


function verify(detector::AbstractEntanglementDetector, param)
    dimH = detector.dimH

    model = Model()
    setMosekParam(model, param)
    # Decision variables
    @variable(model, t >= 0)
    @variable(model, lambda[1:length(detector.purestates)] >= 0)

    # Objective function
    @objective(model, Min, t)
    sumRE = @expression(model, sum(lambda[i] * purestate[:RE] for (i, purestate) in enumerate(detector.purestates)) )
    sumIM = @expression(model, sum(lambda[i] * purestate[:IM] for (i, purestate) in enumerate(detector.purestates)) )
    # constraints

    @constraint(model, [1e-5; vec(sumRE - ( (1 - t) * detector.H[:RE] + t * Diagonal(ones(detector.dimH) / detector.dimH) ))] in MOI.NormOneCone(dimH*dimH + 1) )
    @constraint(model, [1e-5; vec(sumIM - (1 - t) * detector.H[:IM])] in MOI.NormOneCone(dimH*dimH + 1) )
    @constraint(model, sum(lambda) == 1)

    # Solve the model
    status, solverstatus, primalobj, _ = solveMSK(model, param, true)

    # Return results
    print("verification: $(value(t))")
end

function verifyL2(detector::AbstractEntanglementDetector, param)
    dimH = detector.dimH

    model = Model()
    setMosekParam(model, param)
    # Decision variables
    @variable(model, t >= 0)
    @variable(model, lambda[1:length(detector.purestates)] >= 0)

    # Objective function
    @objective(model, Min, t)
    sumRE = @expression(model, sum(lambda[i] * purestate[:RE] for (i, purestate) in enumerate(detector.purestates)) )
    sumIM = @expression(model, sum(lambda[i] * purestate[:IM] for (i, purestate) in enumerate(detector.purestates)) )
    # constraints

    @constraint(model, [t; vcat(vec( sumRE - detector.H[:RE] ), vec(sumIM - detector.H[:IM] )) ] in SecondOrderCone() )
    @constraint(model, sum(lambda) == 1)

    # Solve the model
    status, solverstatus, primalobj, _ = solveMSK(model, param, true)

    # Return results
    print("verification: $(value(t))")
end

function diffNorm(detector::AbstractEntanglementDetector)
    # Global variable to store the average purestate
    global avg_purestate_re = nothing
    global avg_purestate_im = nothing

    # Update running average
    if isnothing(avg_purestate_re)
        avg_purestate_re = detector.purestates[1][:RE]
        avg_purestate_im = detector.purestates[1][:IM]
    else
        n = length(detector.purestates)
        avg_purestate_re = ((n-1) * avg_purestate_re + detector.purestates[end][:RE]) / n
        avg_purestate_im = ((n-1) * avg_purestate_im + detector.purestates[end][:IM]) / n
    end

    if length(detector.purestates) < 2
        return Inf
    end
    last_state = detector.purestates[end]
    prev_state = detector.purestates[end-1]

    diff_re = last_state[:RE] - prev_state[:RE]
    diff_im = last_state[:IM] - prev_state[:IM]

    diff_avg_re = last_state[:RE] - avg_purestate_re
    diff_avg_im = last_state[:IM] - avg_purestate_im
    avg_norm = norm(diff_avg_re) + norm(diff_avg_im)
    return norm(diff_re) + norm(diff_im), avg_norm
end

