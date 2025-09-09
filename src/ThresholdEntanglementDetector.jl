

mutable struct ThresholdEntanglementDetector <: AbstractEntanglementDetector
    H
    dims::Vector{Int64}
    dimH::Int
    nsubs::Int
    M
    cM
    r
    b
    model
    optimizer
    purestates
    substates
    cuts
    presistentinds
    ispersistent
    poolpurestates
    poolsubstates
    poolstats
    round
    function ThresholdEntanglementDetector(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, purestates, substates)
        dimH = reduce(*, dims)
        nsubs = length(dims)
        H = Dict(:RE=>HR, :IM=>HI)
        M = Dict(:RE=>  zeros(AffExpr, 0, 0), :IM=>  zeros(AffExpr, 0, 0))
        cM = Dict(:RE=>  zeros(AffExpr, 0, 0), :IM=>  zeros(AffExpr, 0, 0))
        b =  AffExpr()
        r = AffExpr()
        detector = new(H, dims, dimH, nsubs, M, cM, r, b)
        detector.purestates = copy(purestates)
        detector.substates = copy(substates)
        detector.cuts = []
        detector.presistentinds = []
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
        println("clear non persistent states ", length(detector.presistentinds))
        detector.purestates = [detector.purestates[ind] for ind in detector.presistentinds]
        detector.substates = [detector.substates[ind] for ind in detector.presistentinds]
        detector.ispersistent  = [true] * length(detector.presistentinds)
        detector.presistentinds = [ind for ind in 1:length(detector.presistentinds)]
    end
    detector.cuts = []
end

function normalizationCondition(detector::ThresholdEntanglementDetector, param)
    # normalization condition
    @constraint(detector.model, dot(detector.M[:RE], detector.H[:RE] - Diagonal(ones(detector.dimH) / detector.dimH)) + dot(detector.M[:IM], detector.H[:IM]) <= 1 )
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


function getRank1State(Xvals, nsubs)
    eigenvecs = [[] for i in 1:nsubs]
    for i in 1:nsubs
        vals, vecs = eigen(Xvals[:RE][i] + im * Xvals[:IM][i])
        for j in 1:length(vals)
            if abs(vals[j]) < 1e-6
                continue
            end
            push!(eigenvecs[i], vecs[:, j])
        end
    end
    substatess = []
    for indices in Iterators.product((1:length(eigenvecs[i]) for i in 1:nsubs)...)
        push!(substatess, [eigenvecs[i][indices[i]] for i in 1:nsubs])
    end
    return substatess
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

function detectEntanglementThresholdLiftDiscrete(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    #purestates = getPureStates(HR, HI, dims, param)
    singlerun = false
    clearall = false
    if param.loop == -1
        maxi = 10000000
    elseif param.loop == -2
        maxi = 1
        singlerun = true
        clearall = true
    elseif param.loop == -3
        maxi = 1
        singlerun = true
    else
        maxi = param.loop
    end

    # create the separate problem
    separateproblem = Problem(HR, HI, dims)
    dimH = separateproblem.dimH
    nsubs = length(dims)

    # set data
    multipliers = Dict(:RE=> zeros(dimH, dimH), :IM=> zeros(dimH, dimH))
    ub = 0.5
    glbub = Inf
    glblb = -Inf
    approxub = 1.0
    approxfeas = 0.0
    approxweights = 0.0

    nz_purestates = []
    nz_substates = []
    nz_weights = []

    # create the detector
    detector = ThresholdEntanglementDetector(HR, HI, dims, [], [])

    # add the identity state
    identitystate = Dict(:RE=> [ Matrix(Diagonal(ones(dim))) / dimH for dim in dims], :IM=> [zeros(dim, dim)  for dim in dims])
    addRank1State(detector, identitystate, nothing, nothing, true, false, param.pointsize_bound)

    npersistent = length(detector.substates)
    # add more states to match the point size bound
    purestates_, substates_ = complementStates(dims, param.pointsize_bound, length(detector.substates))
    addBatchStates(detector, purestates_, substates_, false)

    ncomplementpoints = length(detector.substates) - npersistent
    # assign weights
    weights = vcat([2.0 / (2 * npersistent + ncomplementpoints) for _ in 1:npersistent], [1.0 / (2 * npersistent + ncomplementpoints) for _ in 1:ncomplementpoints])
    @assert length(weights) == length(detector.substates) "Weights and substates must have the same length $(length(weights)) != $(length(detector.substates))"

    nnz = max(param.pointsize_bound - param.rank_bound, 0)
    nz_purestates = detector.purestates[end - nnz:end]
    nz_substates = detector.substates[end - nnz:end]

    for i in 1:maxi
        println("Lifting-discretization iteration: $i / $(param.loop)")
        # sort the weights in descending order
        detector.round =  2 * i - 1
        sorted_weights = sort(weights, rev=true)
        sorted_indices = sortperm(weights, rev=true)
        # get sorted substates
        sorted_substates = detector.substates[sorted_indices]
        nfactor = min(length(sorted_weights), param.rank_bound)
        # get the top nfactor substates and weights
        weights = sorted_weights[1:nfactor]
        substates = sorted_substates[1:nfactor]
        # get the nonzero substates
        # normalize the weights
        weights ./= sum(weights)

        purestates, substates, approxub, approxfeas = ALMADMMSolve(detector, dims, HR + im * HI, substates, weights, 1 - ub, multipliers, param, i == 1, singlerun)

        # get the nonzero weights and substates
        clearStates(detector, clearall)

        addBatchStates(detector, purestates, substates, param.lazification)

        addBatchStates(detector, nz_purestates, nz_substates, false)

        detector.round =  2 * i
        # get the lower and upper bounds
        ub, lb, terminate, purestates, substates, weights, multipliers, weights_sum = cuttingPlane(detector, separateproblem, param, 1, singlerun)
        glbub = min(glbub, ub)
        glblb = max(glblb, lb)
        # get nonzero weights and substates
        nz_purestates = []
        nz_substates = []
        nz_weights = []
        approxweights = 0
        ct = 0
        # filter out non-zero weights
        for i in 1:length(weights)
            if weights[i] > 1e-7
                push!(nz_substates, substates[i])
                push!(nz_weights, weights[i])
                push!(nz_purestates, purestates[i])
                ct += 1
            end
        end
        approxweights = weights_sum
        weights = nz_weights
        substates = nz_substates
        # reverse the sign
        multipliers[:RE] = -multipliers[:RE]
        multipliers[:IM] = -multipliers[:IM]
        println("Terminated at iteration $i with number of substates: $ct, glbub: $glbub, glblb: $glblb, approxub: $approxub, approxweights: $approxweights")
        if terminate
            println("Terminated at iteration $i with ub: $ub, lb: $lb")
            break
        end
        if is_time_limit_exceeded(param)
            println("Time limit exceeded, exiting...")
            break
        end
    end
    println("Loop finished $(param.loop)")
    return glbub, glblb, approxub, approxfeas, approxweights
end

function detectEntanglementThresholdDiscrete(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    #purestates = getPureStates(HR, HI, dims, param)
    separateproblem = Problem(HR, HI, dims)
    dimH = separateproblem.dimH
    nsubs = length(dims)
    multipliers = Dict(:RE=> zeros(dimH, dimH), :IM=> zeros(dimH, dimH))
    ub = 0.5

    # create the detector
    detector = ThresholdEntanglementDetector(HR, HI, dims, [], [])

    # add the identity state
    identitystate =  Dict(:RE=> [ Matrix(Diagonal(ones(dim))) / dimH for dim in dims], :IM=> [zeros(dim, dim)  for dim in dims])
    addRank1State(detector, identitystate, nothing, nothing, true, false, param.pointsize_bound)
    npersistent = length(detector.substates)
    # add more states to match the point size bound
    purestates_, substates_ = complementStates(dims, param.pointsize_bound, length(detector.substates))
    addBatchStates(detector, purestates_, substates_, false)


    # get the lower and upper bounds

    ub, lb, purestates, substates, weights, multipliers, weights_sum = cuttingPlane(detector, separateproblem, param, 1)

    return ub, lb, ub, 0.0
end

function detectEntanglementThresholdAlternate(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    #purestates = getPureStates(HR, HI, dims, param)
    separateproblem = Problem(HR, HI, dims)
    dimH = separateproblem.dimH
    nsubs = length(dims)
    multipliers = Dict(:RE=> zeros(dimH, dimH), :IM=> zeros(dimH, dimH))
    ub = 0.5

    # create the detector
    detector = ThresholdEntanglementDetector(HR, HI, dims, [], [])

    # add the identity state
    identitystate =  Dict(:RE=> [ Matrix(Diagonal(ones(dim))) / dimH for dim in dims], :IM=> [zeros(dim, dim)  for dim in dims])
    addRank1State(detector, identitystate, nothing, nothing, true, false, param.pointsize_bound)
    npersistent = length(detector.substates)
    # add more states to match the point size bound
    purestates_, substates_ = complementStates(dims, param.pointsize_bound, length(detector.substates))
    addBatchStates(detector, purestates_, substates_, false)
    weights  = ones(length(detector.substates)) / length(detector.substates)
    # get the lower and upper bounds
    ub, purestates, substates = alternateSolve(dims, HR + im * HI, detector.purestates, detector.substates, weights, param)

    return ub, -Inf64, ub, 0.0
end

function detectEntanglementThresholdPPT(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    #purestates = getPureStates(HR, HI, dims, param)
    H = HR + im * HI
    dimH = reduce(*, dims)
    if dims == [2, 2, 2, 2, 2]
        val, _ = Ket.entanglement_robustness(Matrix(ρ), [8,4], 1; noise="white", ppt=true, inner=false, verbose = true)
        val /= dimH

    elseif dims == [2, 2, 2, 2]
        val, _ = Ket.entanglement_robustness(Matrix(ρ), [4,4], 2; noise="white", ppt=true, inner=false, verbose = true)
        val /= dimH
    elseif dims == [2, 2, 2]
        val, _ = Ket.entanglement_robustness(Matrix(ρ), [4,2], 6; noise="white", ppt=true, inner=false, verbose = true)
        val /= dimH
    end
    lb = val
    return lb
end

function detectEntanglementThresholdHybridSingle(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    #purestates = getPureStates(HR, HI, dims, param)
    singlerun = false
    clearall = false
    maxi = 10000000
    singlerun = false
    clearall = false

    # create the separate problem
    separateproblem = Problem(HR, HI, dims)
    dimH = separateproblem.dimH
    nsubs = length(dims)

    # set data
    multipliers = Dict(:RE=> zeros(dimH, dimH), :IM=> zeros(dimH, dimH))
    ub = 0.5
    glbub = Inf
    glblb = -Inf
    approxub = 1.0
    approxfeas = 0.0
    approxweights = 0.0

    nz_purestates = []
    nz_substates = []
    nz_weights = []

    # create the detector
    detector = ThresholdEntanglementDetector(HR, HI, dims, [], [])

    # add the identity state
    identitystate =  Dict(:RE=> [ Matrix(Diagonal(ones(dim))) / dimH for dim in dims], :IM=> [zeros(dim, dim)  for dim in dims])
    addRank1State(detector, identitystate, nothing, nothing, true, false, param.pointsize_bound)
    npersistent = length(detector.substates)
    # add more states to match the point size bound
    purestates_, substates_ = complementStates(dims, param.pointsize_bound, length(detector.substates))
    addBatchStates(detector, purestates_, substates_, false)
    ncomplementpoints = length(detector.substates) - npersistent
    # assign weights
    weights = vcat([2.0 / (2 * npersistent + ncomplementpoints) for _ in 1:npersistent], [1.0 / (2 * npersistent + ncomplementpoints) for _ in 1:ncomplementpoints])
    @assert length(weights) == length(detector.substates) "Weights and substates must have the same length $(length(weights)) != $(length(detector.substates))"

    nnz = max(param.pointsize_bound - param.rank_bound, 0)
    nz_purestates = detector.purestates[end - nnz:end]
    nz_substates = detector.substates[end - nnz:end]

    for i in 1:maxi
        println("Lifting-discretization iteration: $i / $(param.loop)")
        # sort the weights in descending order
        sorted_weights = sort(weights, rev=true)
        sorted_indices = sortperm(weights, rev=true)
        # get sorted substates
        sorted_substates = detector.substates[sorted_indices]
        sorted_purestates = detector.purestates[sorted_indices]
        nfactor = min(length(sorted_weights), param.rank_bound)
        # get the top nfactor substates and weights
        weights = sorted_weights[1:nfactor]
        substates = sorted_substates[1:nfactor]
        purestates = sorted_purestates[1:nfactor]
        # get the nonzero substates
        # normalize the weights
        weights ./= sum(weights)

        ub, purestates, substates = alternateSolve(dims, HR + im * HI, purestates, substates, weights, param, i == 1)
        glbub = min(glbub, ub)

        # get the nonzero weights and substates
        clearStates(detector, clearall)
        addBatchStates(detector, purestates, substates, param.lazification)
        addBatchStates(detector, nz_purestates, nz_substates)

        # get the lower and upper bounds
        ub, lb, terminate, purestates, substates, weights, multipliers, weights_sum = cuttingPlane(detector, separateproblem, param, 1, singlerun)
        glbub = min(glbub, ub)
        glblb = max(glblb, lb)
        # get nonzero weights and substates
        nz_purestates = []
        nz_substates = []
        nz_weights = []
        approxweights = 0
        ct = 0
        # filter out non-zero weights
        for i in 1:length(weights)
            if weights[i] > 1e-7
                push!(nz_substates, substates[i])
                push!(nz_weights, weights[i])
                push!(nz_purestates, purestates[i])
                ct += 1
            end
        end

        approxweights = weights_sum
        weights = nz_weights
        substates = nz_substates
        # reverse the sign
        multipliers[:RE] = -multipliers[:RE]
        multipliers[:IM] = -multipliers[:IM]
        println("Terminated at iteration $i with number of substates: $ct, glbub: $glbub, glblb: $glblb, approxub: $approxub, approxweights: $approxweights")
        if terminate
            println("Terminated at iteration $i with ub: $ub, lb: $lb")
            break
        end
        if is_time_limit_exceeded(param)
            println("Time limit exceeded, exiting...")
            break
        end
    end

    return glbub, glblb, glbub, 0.0, approxweights
end


function detectEntanglementThresholdLiftDual(HR::Matrix{Float64}, HI::Matrix{Float64}, dims::Vector{Int64}, param::Param)
    #purestates = getPureStates(HR, HI, dims, param)
    singlerun = false
    clearall = false
    if param.loop == -1
        maxi = 10000000
    elseif param.loop == -2
        maxi = 1
        singlerun = true
        clearall = true
    elseif param.loop == -3
        maxi = 1
        singlerun = true
    else
        maxi = param.loop
    end

    # create the separate problem
    separateproblem = Problem(HR, HI, dims)
    dimH = separateproblem.dimH
    nsubs = length(dims)

    # set data
    multipliers = 1.0
    ub = 0.5
    glbub = Inf
    glblb = -Inf
    approxub = 1.0
    approxfeas = 0.0
    approxweights = 0.0

    nz_purestates = []
    nz_substates = []
    nz_weights = []

    # create the detector
    detector = ThresholdEntanglementDetector(HR, HI, dims, [], [])

    # add the identity state
    identitystate = Dict(:RE=> [ Matrix(Diagonal(ones(dim))) / dimH for dim in dims], :IM=> [zeros(dim, dim)  for dim in dims])
    addRank1State(detector, identitystate, nothing, nothing, true, false, param.pointsize_bound)

    npersistent = length(detector.substates)
    # add more states to match the point size bound

    purestates_, substates_ = complementStates(dims, param.pointsize_bound, length(detector.substates))
    addBatchStates(detector, purestates_, substates_, false)

    ncomplementpoints = length(detector.substates) - npersistent
    # assign weights
    weights = vcat([2.0 / (2 * npersistent + ncomplementpoints) for _ in 1:npersistent], [1.0 / (2 * npersistent + ncomplementpoints) for _ in 1:ncomplementpoints])
    @assert length(weights) == length(detector.substates) "Weights and substates must have the same length $(length(weights)) != $(length(detector.substates))"

    nnz = max(param.pointsize_bound - param.rank_bound, 0)
    nz_purestates = detector.purestates[end - nnz:end]
    nz_substates = detector.substates[end - nnz:end]

    sorted_weights = sort(weights, rev=true)
    sorted_indices = sortperm(weights, rev=true)
    # get sorted substates
    sorted_substates = detector.substates[sorted_indices]
    nfactor = min(length(sorted_weights), param.rank_bound)
    # get the top nfactor substates and weights
    weights = sorted_weights[1:nfactor]
    substates = sorted_substates[1:nfactor]
    # get the nonzero substates
    # normalize the weights
    weights ./= sum(weights)

    purestates, substates, approxlb, approxfeas = dualALMSolve(detector, dims, HR + im * HI, substates, weights, multipliers, param, true, singlerun)

    return 0, 0, approxlb, approxfeas, 0.0
end
