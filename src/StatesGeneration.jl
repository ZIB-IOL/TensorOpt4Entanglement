function cuttingPlane(detector::AbstractEntanglementDetector, separateproblem, param::Param, effortlevel = 0, singlerun = false)
    initialLPRelaxation(detector, param)
    print("LP relaxation created--\n")
    iter = 0
    nlmocall = 0
    nlazycall = 0
    nlazyfound = 0
    poolsize = length(detector.poolsubstates)
    Mout = Dict(:RE=>zeros(detector.dimH, detector.dimH), :IM=>zeros(detector.dimH, detector.dimH))
    valueb = 1.0
    primalobj = Inf
    primalobjprev = Inf
    dualobj = -Inf
    masterconverge = false
    maxfail = 500
    fails = 0
    weights = []
    weghts_sum = 0.0
    terminate = false
    while true
        primalobjprev = primalobj
        # check time limit
        is_time_limit_exceeded(param)
        # check is the last period
        islast = param.is_last
        needincrease = false
        if !islast && is_time_limit_last(param)
            println("Time limit is almost reached")
            needincrease = true
            if param.lazification
                poolAdd(detector, param)
            end
        end
        print("solve--\n")
        status, solverstatus, primalobj, _ = solveMSK(detector.model, param, false)
        print("end solve-- $status $solverstatus\n")
        is_time_limit_exceeded(param)
        if needincrease
            increase_last_time_limit(param)
            islast = true
        end
        if status == RelaxOptimal || status == RelaxFeasible
            Mout[:RE] = value.(detector.M[:RE])
            Mout[:IM] = value.(detector.M[:IM])
            valueb = value(detector.b)
            weights = [abs(dual(cut))  for cut in detector.cuts]
            weights_sum = sum(weights)
            println(" weights_sum: $(weights_sum)")
            weights .= weights ./ weights_sum
            updateProblem(detector, separateproblem, primalobj, param)
            isEarlyStopping = earlyStopping(detector, primalobj, param)
            if isEarlyStopping || singlerun
                break
            end
            separateproblem.H = Mout
            separateproblem.Hout = Mout
            separateproblem.cutoffbound = valueb
            itereffortlevel = 0
            if islast
                itereffortlevel = 2
            end
            if is_time_limit_exceeded(param)
                break
            end
            print("itereffortlevel: $(itereffortlevel)\n")
            terminate = false
            lmocall = true
            if lmocall
                cutactiveness, cutdualactiveness, state, Xvals = separate!(separateproblem, param, itereffortlevel, false)
                terminate, addstate, newdualobj = checkTerminationGap(detector, primalobj, cutactiveness, cutdualactiveness, valueb, dualobj, param)
                dualobj = max(newdualobj, dualobj)
                nlmocall += 1
            end
            if param.log_level > 0
                print("iteration: $(iter),  #states: $(length(detector.purestates)), valueb: $(valueb), activeness: $(cutactiveness), primalobj: $(primalobj), dualobj: $(dualobj)\n")
            end
            if param.maxrounds >=0 && !param.is_last && nlmocall >= min(param.maxrounds, 2 * detector.dimH * detector.dimH + 1 )
                print("max rounds reached and not in last state generation\n")
                break
            end
            if terminate || masterconverge
                break
            elseif addstate
                #addState(detector, state)
                addRank1PrincipleState(detector, Xvals, Mout, valueb, false, param.lazification)
                #check(detector.substates, detector.dims)
                #@assert( length(detector.cuts) == length(detector.substates) )
            else
                print("fail to find a state\n")
                break
            end
            iter += 1
        else
            weights = ones(length(detector.cuts))
            weights_sum = sum(weights)
            weights .= weights ./ weights_sum
            break
        end
    end
    #check(detector.substates, detector.dims)
    if param.lazification
        detector.poolpurestates = []
        detector.poolsubstates = []
        detector.poolstats = []
    end
    return primalobj, dualobj, terminate, detector.purestates, detector.substates, weights, Mout, weghts_sum
end


function cuttingPlaneRestart(detector::AbstractEntanglementDetector, separateproblem, param::Param, effortlevel = 0)
    initialLPRelaxation(detector, param)
    iter = 0
    Mout = Dict(:RE=>zeros(detector.dimH, detector.dimH), :IM=>zeros(detector.dimH, detector.dimH))
    valueb = 1.0
    primalobj = Inf
    primalobjprev = Inf
    dualobj = -Inf
    masterconverge = false
    maxfail = 3
    restartiter =  3 * detector.dimH * detector.dimH * 2
    fails = 0
    enoughrun = false
    maxrestart = 15000
    while iter <= param.maxrounds
        primalobjprev = primalobj
        status, solverstatus, primalobj, _ = solveMSK(detector.model, param, true)
        if status == RelaxOptimal || status == RelaxFeasible
            Mout[:RE] = value.(detector.M[:RE])
            Mout[:IM] = value.(detector.M[:IM])
            valueb = value(detector.b)
            if primalobj > primalobjprev - param.master_obj_tol
                fails += 1
            else
                fails = 0
            end
            if (iter + 1) % restartiter == 0 &&  iter < maxrestart
                enoughrun = true
            end
            if enoughrun && fails >= maxfail
                activepurestates = []
                print("----restart ---- \n")
                for state in detector.purestates
                    if abs(dot(Mout[:RE], real(state)) + dot(Mout[:IM], imag(state)) - valueb) < param.tol
                        push!(activepurestates, state)
                    end
                end
                dimH = detector.dimH
                push!(activepurestates, Hermitian(Matrix( Diagonal(ones(dimH)) / dimH)) )
                detector.purestates = activepurestates
                initialLPRelaxation(detector, param)
                primalobj = Inf
                primalobjprev = Inf
                iter += 1
                enoughrun = false
                continue
            end
            updateProblem(detector, separateproblem, primalobj, param)
            isEarlyStopping = earlyStopping(detector, primalobj, param)
            if isEarlyStopping
                return
            end
            separateproblem.H = Mout
            separateproblem.Hout = Mout
            separateproblem.cutoffbound = valueb
            itereffortlevel = 0
            if effortlevel == 1
                itereffortlevel =  param.maxrounds ? effortlevel : 0
            elseif effortlevel == 2
                if iter == param.maxrounds
                    itereffortlevel = 2
                else
                    itereffortlevel = 1
                end
            end
            needsglobalobbt = iter >= 5 * param.freq_globalobbt && iter % param.freq_globalobbt == 0 && itereffortlevel < 2
            cutactiveness, cutdualactiveness, state, Xvals = separate!(separateproblem, param, itereffortlevel, needsglobalobbt)
            terminate, addstate, newdualobj = checkTerminationGap(detector, primalobj, cutactiveness, cutdualactiveness, valueb, dualobj, param)
            dualobj =  max(newdualobj, dualobj)
            if param.log_level > 0
                print("iteration: $(iter), #states: $(length(detector.purestates)), valueb: $(valueb), activeness: $(cutactiveness),  primalobj: $((1+param.extrascale)*primalobj - param.extrascale), dualobj: $((1+param.extrascale)*dualobj - param.extrascale)\n")
            end
            if terminate || masterconverge
                return
            elseif addstate
                addState(detector, state)
            else
                error("fail to find a state\n")
            end
            iter += 1
        else
            error("Error: LP Relaxation failed $(status) $(solverstatus)\n")
        end
    end
end

function cuttingPlaneStablized(detector::AbstractEntanglementDetector, separateproblem, param::Param, effortlevel = 0)
    initialLPRelaxation(detector, param)
    iter = 0
    Mout = Dict(:RE=>zeros(detector.dimH, detector.dimH), :IM=>zeros(detector.dimH, detector.dimH))
    valueb = 1.0
    primalobj = Inf
    primalobjprev = Inf
    dualobj = -Inf
    Min = Dict(:RE=> detector.H[:RE] - Diagonal(ones(detector.dimH) / detector.dimH),
               :IM=> detector.H[:IM] )
    Minnorm = sqrt(dot(Min[:RE],Min[:RE]) + dot(Min[:IM],Min[:IM]))
    Min[:RE] /= Minnorm
    Min[:IM] /= Minnorm
    deltaalpha = 1 / param.nalpha
    while iter <= param.maxrounds
        primalobjprev = primalobj
        status, solverstatus, primalobj, _ = solveMSK(detector.model, param, true)
        if status == RelaxOptimal || status == RelaxFeasible
            Mout[:RE] = value.(detector.M[:RE])
            Mout[:IM] = value.(detector.M[:IM])
            Moutnorm = sqrt(dot(Mout[:RE],Mout[:RE]) + dot(Mout[:IM],Mout[:IM]))
            valueb = value(detector.b)
            updateProblem(detector, separateproblem, primalobj, param)
            isEarlyStopping = earlyStopping(detector, primalobj, param)
            if isEarlyStopping
                return
            end
            # stablization iteration
            for i in 1:param.nalpha
                # effort level
                itereffortlevel = 0
                if effortlevel == 1
                    itereffortlevel = iter == param.maxrounds ? 2 : 0
                elseif effortlevel == 2
                    if iter == param.maxrounds
                        itereffortlevel = 2
                    else
                        itereffortlevel = 1
                    end
                end
                # is obbt needed?
                needsglobalobbt = iter >= 100 * param.freq_globalobbt && iter % param.freq_globalobbt == 0 && itereffortlevel < 2
                # go to normal separation
                if iter == param.maxrounds
                    i = param.nalpha
                end
                if i < param.nalpha
                    alpha = 1 - deltaalpha * i
                    # set Msepa
                    Msepa = Dict()
                    # convex combination
                    Msepa[:RE] = alpha * Min[:RE] + (1 - alpha) * Mout[:RE] / Moutnorm
                    Msepa[:IM] = alpha * Min[:IM] + (1 - alpha) * Mout[:IM] / Moutnorm
                    separateproblem.H = Msepa
                    separateproblem.Hout = Mout
                    separateproblem.cutoffbound = valueb
                    _, _, state, Xvals = separate!(separateproblem, param, itereffortlevel, needsglobalobbt)
                    cutactiveness = dot(Mout[:RE], real(state)) + dot(Mout[:IM], imag(state))
                    if param.log_level > 0
                        print("iteration: $(iter), i: $(i), valueb: $(valueb), activeness: $(cutactiveness), primalobj: $(primalobj), dualobj: $(dualobj)\n")
                    end
                    addRank1State(detector, Xvals, Mout, valueb)
                    if cutactiveness > valueb
                        break
                    end
                else
                    separateproblem.H = Mout
                    separateproblem.Hout = Mout
                    separateproblem.cutoffbound = valueb
                    cutactiveness, cutdualactiveness, _, Xvals = separate!(separateproblem, param, itereffortlevel, needsglobalobbt)
                    terminate, addstate, newdualobj = checkTerminationGap(detector, primalobj, cutactiveness, cutdualactiveness, valueb, dualobj, param)
                    dualobj =  max(newdualobj, dualobj)
                    if param.log_level > 0
                        print("iteration: $(iter), #states: $(length(detector.purestates)),  valueb: $(valueb), activeness: $(cutactiveness), primalobj: $(primalobj), dualobj: $(dualobj)\n")
                    end
                    if terminate
                        return
                    elseif addstate
                        #addState(detector, state)
                        addRank1State(detector, Xvals, Mout, valueb)
                    else
                        error("fail to find a state\n")
                    end
                    iter += 1
                    break
                end
            end
        else
            error("Error: LP Relaxation failed $(status) $(solverstatus)\n")
        end
    end
end

function getPureStates(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    epsilon = 1e-7
    dimH = prod(dims)
    dims = Tuple(dims)
    H = Hermitian(HR + im * HI)
    σ = Matrix{ComplexF64}(I, dimH, dimH)./prod(dimH)
    lmo=ED.AlternatingSeparableLMO(float(real(ComplexF64)), dims)
    noise =ED.correlation_tensor(σ, dims, lmo.matrix_basis)
    noise_atol = 1e-4
    max_iteration = 10^6
    res = ED.separable_distance(H, dims; noise_mixture = true, noise = noise, verbose = true, noise_atol = noise_atol, max_iteration = max_iteration , epsilon = epsilon)
    distanceupperbound = 1 - res.primal
    active_set = res.active_set
    purestates = [ ED.density_matrix(point[2]) for point in active_set if point[1] > param.tol]
    purestates = [ purestate / tr(purestate) for purestate in purestates]
    push!(purestates, Hermitian(Matrix( Diagonal(ones(dimH)) / dimH))  )
    return purestates
end

function orthogonalStates(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    nsubs = length(dims)
    substates = [[] for i in 1:nsubs]
    purestates = []

    # For each dimension
    for (i, d) in enumerate(dims)
        # Create d orthogonal vectors
        basis = [zeros(ComplexF64, d) for _ in 1:d]
        for i in 1:d
            basis[i][i] = 1.0
        end

        # Convert each vector to density matrix and add to purestates
        for vec in basis
            ρ = vec * vec' # Create density matrix from pure state
            ρ = Hermitian(ρ / tr(ρ)) # Normalize and ensure Hermitian
            push!(substates[i], ρ)
        end
    end

    # Add maximally mixed state
    ct = 0
    for indices in Iterators.product((1:length(substates[i]) for i in 1:nsubs)...)
        substate = [substates[i][indices[i]] for i in 1:nsubs]
        state = foldl(kron, substate)
        push!(purestates, state)
    end

    return purestates
end

function getPureStates_(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    epsilon = 1e-5
    tuple_dims = Tuple(dims)
    H = Hermitian(HR + im * HI)
    ent_proof = false
    sep_search = false
    epsilon = 1e-7          # control primal, the smaller the better for sep_bound, approximately, epsilon < 2 noise_atol^2 / D^2
    max_iteration = 10^5    # control time
    callback_iter = 10^5
    noise_atol = 1e-4       # control the accuracy of bounds
    verbose = true
    max_length = 10^6       # when ent_proof is true, control the time
    min_η = 1 - 0.01        # when ent_proof is true, control the accuracy for ent_bound
    res = ED.white_noise_robustness(H, tuple_dims; ent_proof, sep_search, epsilon, max_iteration, noise_atol, verbose, callback_iter, max_length, min_η)
    distanceupperbound = res.primal
    if distanceupperbound < param.master_obj_tol
        print("the state is not entangled\n")
        return
    end
    active_set = res.active_set
    purestates = [ ED.density_matrix(point[2]) for point in active_set if point[1] > param.tol]
    purestates = [ purestate / tr(purestate) for purestate in purestates]
    dimH = reduce(*, dims)
    push!(purestates, Hermitian(Matrix( Diagonal(ones(dimH)) / dimH))  )
    return purestates
end