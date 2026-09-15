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
"""
cuttingPlane(detector::AbstractEntanglementDetector, separateproblem, param::Param,
             effortlevel = 0, singlerun = false) =
    withPhase(:cp) do
        cuttingPlane_(detector, separateproblem, param, effortlevel, singlerun)
    end

function cuttingPlane_(detector::AbstractEntanglementDetector, separateproblem, param::Param, effortlevel = 0, singlerun = false)
    initialLPRelaxation(detector, param)
    print("LP relaxation created--\n")
    trace = cpTraceSink()
    iter = 0
    nlmocall = 0
    Mout = Dict(:RE=>zeros(detector.dimH, detector.dimH), :IM=>zeros(detector.dimH, detector.dimH))
    valueb = 1.0
    primalobj = Inf
    dualobj = -Inf
    masterconverge = false
    weights = []
    weights_sum = 0.0
    terminate = false
    while true
        # check time limit
        isTimeLimitExceeded(param)
        # check is the last period
        islast = param.is_last
        needincrease = false
        if !islast && isTimeLimitNearlyReached(param)
            println("Time limit is almost reached")
            needincrease = true
            if param.lazification
                poolAdd(detector, param)
            end
        end
        print("solve--\n")
        notePhaseModel!(:cp, detector.model; nnz = true)
        status, solverstatus, primalobj, _ = solveMSK(detector.model, param, false)
        print("end solve-- $status $solverstatus\n")
        isTimeLimitExceeded(param)
        if needincrease
            extendTimeLimit!(param)
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
            if isTimeLimitExceeded(param)
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
            # b_lower is the LMO's own lower bound for this round; lb_relx = ub_relx + b_lower
            traceRow!(trace, iter, param.is_last, primalobj, dualobj, newdualobj - primalobj, length(detector.purestates))
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
                addRank1PrincipleState(detector, Xvals, Mout, valueb, false, param.lazification)
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
    traceClose!(trace)
    if param.lazification
        detector.poolpurestates = []
        detector.poolsubstates = []
        detector.poolstats = []
    end
    return primalobj, dualobj, terminate, detector.purestates, detector.substates, weights, Mout, weights_sum
end

