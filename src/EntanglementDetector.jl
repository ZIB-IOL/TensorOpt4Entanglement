# Entanglement Detector data
mutable struct EntanglementDetector
    H
    dims::Vector{Int64}
    dimH::Int
    nsubs::Int
    M
    b
    model
    cuts
    purestates
    substates

    function EntanglementDetector(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param, purestates)
        dimH = reduce(*, dims)
        nsubs = length(dims)
        H = Dict(:RE=>HR, :IM=>HI)
        M = Dict(:RE=>  zeros(AffExpr, 0, 0), :IM=>  zeros(AffExpr, 0, 0))
        b =  AffExpr()
        cuts = Set([])
        substates = []
        entanglementdetector = new(H, dims, dimH, nsubs, M, b, cuts, [], substates)
        entanglementdetector.purestates = copy(purestates)
        entanglementdetector.substates = []
        return entanglementdetector
    end
end

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

function importStates(H, dims, param::Param)
    epsilon = 1e-7
    tuple_dims = Tuple(dims)
    if param.heur_FW_maxiters == -1
        σ, v, primal, active_set, lmo = ED.separable_distance(Hermitian(H), tuple_dims; verbose = param.log_level > 1, epsilon)
    else
        σ, v, primal, active_set, lmo = ED.separable_distance(Hermitian(H), tuple_dims; verbose = param.log_level > 1, max_iteration = param.heur_FW_maxiters, epsilon)
    end
    return primal, active_set
end


function initialLPRelaxation(detector::EntanglementDetector, param)
    dimH = detector.dimH

    model = Model()
    setMosekParam(model, param, true)
    @variable(model, MI[1:dimH, 1:dimH])
    @variable(model, MR[1:dimH, 1:dimH])
    @variable(model, b)
    detector.M[:RE] = MR
    detector.M[:IM] = MI
    detector.b = b
    l1normsize = 2 * detector.dimH ^ 2 + 1
    # normalization condition
    @constraint(model, [1; vcat(vec(detector.M[:RE]), vec(detector.M[:IM])) ] in MOI.NormOneCone(l1normsize) )
    @objective(model, Max, dot(detector.M[:RE], detector.H[:RE]) + dot(detector.M[:IM], detector.H[:IM]) - detector.b )

    for state in detector.purestates
        @constraint(model, dot(detector.M[:RE], state[:RE]) + dot(detector.M[:IM], state[:IM]) - detector.b <= 0)
    end

    detector.model = model

end


function addState(detector, state)
    @constraint(detector.model, dot(detector.M[:RE], real(state)) + dot(detector.M[:IM], imag(state)) - detector.b <= 0)
    push!(detector.substates, Dict(:RE=>real(state), :IM=>imag(state)))
end


function addRank1State_(detector, Xvals, valM, valueb)
    nsubs = detector.nsubs
    substates = [[] for i in 1:nsubs]
    for i in 1:nsubs
        vals, vecs = eigen(Xvals[:RE][i] + im * Xvals[:IM][i])
        for j in 1:length(vals)
            if abs(vals[j]) < 1e-6
                continue
            end
            push!(substates[i], vecs[:, j] * vecs[:, j]')
        end
    end
    ct = 0
    for indices in Iterators.product((1:length(substates[i]) for i in 1:nsubs)...)
        substate = [substates[i][indices[i]] for i in 1:nsubs]
        state = foldl(kron, substate)
        violation = dot(valM[:RE], real(state)) + dot(valM[:IM], imag(state)) - valueb
        if violation > 0
            print((violation,valueb),"\n")
            ct += 1
            @constraint(detector.model, dot(detector.M[:RE], real(state)) + dot(detector.M[:IM], imag(state)) - detector.b <= 0)
            push!(detector.substates, substate)
        end
    end
end

function cleanCuts(detector)
    ct = 0
    values = [value(c) for c in detector.cuts]
    for (i, c) in enumerate(detector.cuts)
        if values[i] < -1e-5
            delete(detector.model, c)
            delete!(detector.cuts, c)
            ct += 1
        end
    end
    print("delete $(ct) cuts\n")
end

function cuttingPlane_(detector, separateproblem, param::Param)
    initialLPRelaxation(detector, param)
    iter = 0
    valM = Dict(:RE=>zeros(detector.dimH, detector.dimH), :IM=>zeros(detector.dimH, detector.dimH))
    valueb = 1.0
    while iter <= param.maxrounds
        status, _, primalobj, _ = solveMSK(detector.model, param, true)
        if status == RelaxOptimal || status == RelaxFeasible
            valM[:RE] = value.(detector.M[:RE])
            valM[:IM] = value.(detector.M[:IM])
            valueb = value(detector.b)
            if primalobj < -param.master_obj_tol
                print("the state is not entangled: $(primalobj)\n")
                return
            end
            separateproblem.H = valM
            separateproblem.cutoffbound = valueb
            cutactiveness, cutdualactiveness, state, Xvals = separate(separateproblem, param)
            if param.log_level > 0
                cutactiveness_ = dot(valM[:RE], real(state)) + dot(valM[:IM], imag(state))
                print("iteration: $(iter), obj: $(primalobj), activeness: $(cutactiveness), activeness_: $(cutactiveness_), cutoff: $(valueb)\n")
            end
            if cutactiveness > valueb
                #addState(detector, state)
                addRank1State(detector, Xvals, valM, valueb)
                print("add a state \n")
            elseif cutdualactiveness < valueb
                print("the state is entangled\n")
                return
            else
                error("fail to find a state\n")
            end
            iter += 1
        else
            error("Error: LP Relaxation failed")
        end
    end
end


function cuttingPlaneStablized(detector, separateproblem, param::Param)
    initialLPRelaxation(detector, param)
    valMin = Dict(:RE=>ones(detector.dimH, detector.dimH), :IM=>zeros(detector.dimH, detector.dimH))
    scale = tr(valMin[:RE])
    valMin[:RE] /= scale
    valbin = 2.0 / scale
    valMout = Dict(:RE=>zeros(detector.dimH, detector.dimH), :IM=>zeros(detector.dimH, detector.dimH))
    valbout = 1.0
    valMsepa = Dict(:RE=>zeros(detector.dimH, detector.dimH), :IM=>zeros(detector.dimH, detector.dimH))
    valbsepa = 1.0
    alpha  = param.inout_alpha
    lambda = param.inout_lambda
    maxreset = param.inout_maxreset
    prevobj = Inf
    objimprovefails = 0
    inupdateiter = 0
    kellymode = true
    iter = 0
    skipopt = false
    while iter <= param.maxrounds
        status = RelaxUnsolve
        primalobj = prevobj
        if !skipopt
            status, _, primalobj, _ = solveMSK(detector.model, param, true)
        end
        if skipopt || status == RelaxOptimal || status == RelaxFeasible
            # inside the polytope
            if primalobj < -param.master_obj_tol
                print("the state is not entangled: $(primalobj)\n")
                return
            end
            if !skipopt
                # get out point
                valMout[:RE] = value.(detector.M[:RE])
                valMout[:IM] = value.(detector.M[:IM])
                valbout = value(detector.b)
            end
            # try mode switch, when progress is slow
            if primalobj - prevobj > - param.master_obj_tol
                objimprovefails += 1
                prevobj = primalobj
                if objimprovefails >= maxreset
                    if kellymode
                        # restart, if we are not in kelly mode
                        lambda = param.inout_lambda
                        kellymode = false
                        valMin = Dict(:RE=>ones(detector.dimH, detector.dimH), :IM=>zeros(detector.dimH, detector.dimH))
                        valMin[:RE] /= scale
                        valbin = 2.0 / scale
                        prevobj = Inf
                        inupdateiter = 0
                        #cleanCuts(detector)
                    else
                        # switch to kelly mode
                        lambda = 1.0
                        kellymode = true
                        inupdateiter = 0
                        prevobj = Inf
                    end
                    objimprovefails = 0
                    print("fail to improve the objective, switch to kelly mode: $(kellymode)\n")
                end
            else
                prevobj = primalobj
            end
            if kellymode
                # get separation point
                valMsepa[:RE] = valMout[:RE]
                valMsepa[:IM] = valMout[:IM]
                valbsepa = valbout
            else
                # get separation point
                valMsepa[:RE] .= (1 - lambda) * valMin[:RE] +  lambda * valMout[:RE]
                valMsepa[:IM] .= (1 - lambda) * valMin[:IM] + lambda * valMout[:IM]
                valbsepa = (1 - lambda) * valbin + lambda * valbout
            end
            # set separation problem
            separateproblem.H = valMsepa
            separateproblem.cutoffbound = valbsepa
            cutactiveness, cutdualactiveness, state, Xvals = separate(separateproblem, param)
            if param.log_level > 0
                print("iteration: $(iter), obj: $(primalobj), prevobj: $(prevobj), activeness: $(cutactiveness), cutoff: $(valbsepa), kellymode: $(kellymode), objimprovefails: $(objimprovefails), inupdateiter: $(inupdateiter)\n")
            end
            if cutactiveness > valbsepa
                # find a cut
                #addState(detector, state)
                addRank1State(detector, Xvals, separateproblem.H, separateproblem.cutoffbound)
                print("add a state \n")
                skipopt = false
                inupdateiter = 0
            else
                if kellymode
                    if cutdualactiveness < valueb
                        # outside the polytope
                        print("the state is entangled\n")
                        return
                    else
                        error("fail to find a state\n")
                    end
                else
                    # update in point
                    valMin[:RE] .= alpha * valMin[:RE] + (1 - alpha) * valMout[:RE]
                    valMin[:IM] .= alpha * valMin[:IM] + (1 - alpha) * valMout[:IM]
                    valbin = alpha * valbin + (1 - alpha) * valbout
                    print("update the in state\n")
                    inupdateiter += 1
                    skipopt = true
                    if inupdateiter >= maxreset
                        # too many in updates, try separatio on the out point
                        separateproblem.H = valMout
                        separateproblem.cutoffbound = valbout
                        cutactiveness, cutdualactiveness, state, Xvals = separate(separateproblem, param)
                        if cutactiveness > valbsepa
                            #addState(detector, state)
                            # add a cut
                            addRank1State(detector, Xvals, separateproblem.H, separateproblem.cutoffbound)
                            print("in: add a state \n")
                            inupdateiter = 0
                            skipopt = false
                        elseif cutdualactiveness < valbout
                            # outside the polytope
                            print("in: the state is entangled\n")
                            return
                        else
                            error("fail to find a state\n")
                        end
                    end
                end
            end
            iter += 1
        else
            error("Error: LP Relaxation failed")
        end
    end
end


function cuttingRestart(detector, separateproblem, param::Param)
    initialLPRelaxation(detector, param)
    iter = 0
    valM = Dict(:RE=>zeros(detector.dimH, detector.dimH), :IM=>zeros(detector.dimH, detector.dimH))
    valueb = 1.0
    while iter <= param.maxrounds
        status, _, primalobj, _ = solveMSK(detector.model, param, true)
        if status == RelaxOptimal || status == RelaxFeasible
            valM[:RE] = value.(detector.M[:RE])
            valM[:IM] = value.(detector.M[:IM])
            valueb = value(detector.b)
            if primalobj < -param.master_obj_tol
                print("the state is not entangled: $(primalobj)\n")
                return
            end
            separateproblem.H = valM
            separateproblem.cutoffbound = valueb
            cutactiveness, cutdualactiveness, state, Xvals = separate(separateproblem, param)
            if param.log_level > 0
                cutactiveness_ = dot(valM[:RE], real(state)) + dot(valM[:IM], imag(state))
                print("iteration: $(iter), obj: $(primalobj), activeness: $(cutactiveness), activeness_: $(cutactiveness_), cutoff: $(valueb)\n")
            end
            if cutactiveness > valueb
                #addState(detector, state)
                addRank1State(detector, Xvals, valM, valueb)
                print("add a state \n")
            elseif cutdualactiveness < valueb
                print("the state is entangled\n")
                return
            else
                error("fail to find a state\n")
            end
            iter += 1
        else
            error("Error: LP Relaxation failed")
        end
        if iter % 50 == 0
            initialLPRelaxation(detector, param)
            H_ = valM[:RE] + im*valM[:IM]
            substates = copy(detector.substates)
            detector.substates = []
            for substate in substates
                #cutvalbefore = dot(valM[:RE], real(state)) + dot(valM[:IM], imag(state))
                newsubstate, _ = ManOptSolve(detector.dims, detector.nsubs, H_, detector.dimH, substate, param)
                #cutvalafter = dot(valM[:RE], real(newstate)) + dot(valM[:IM], imag(newstate))
                addRank1State(detector, newsubstate, valM, valueb)
            end
        end
    end
end

