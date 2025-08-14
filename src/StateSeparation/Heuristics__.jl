function constructFullSol(Xvals)
    Hbar = 1.0
    nsubs = length(Xvals[:RE])
    for i in 1:nsubs
        if i == 1
            Hbar = Xvals[:RE][i] + im * Xvals[:IM][i]
        else
            Hbar = kron(Hbar, Xvals[:RE][i] + im * Xvals[:IM][i])
        end
    end
    return Hbar
end

function validate(Xvals)
    for i in 1:length(Xvals[:RE])
        X= Xvals[:RE][i] + im * Xvals[:IM][i]
        if !(abs(tr(X) - 1.0) < 1e-4)
            print(i, " ", abs(tr(X) - 1.0), " ", tr(X), "\n")
        end
        @assert abs(tr(X) - 1.0) < 1e-4
        eigvals = eigen(X).values
        if ! all(x -> real(x) >= -1e-4, eigvals)
            print(i, " ", eigvals, "\n")
        end
        @assert all(x -> real(x) >= -1e-4, eigvals)
    end
end

function diff(Xvals1, Xvals2)
    difference_norm = norm([Xvals1[:RE][i] - Xvals2[:RE][i] + im * (Xvals1[:IM][i] - Xvals2[:IM][i]) for i in 1:length(Xvals1[:RE])])
end



function restrict(H, Xvals, nsubs, dims, j)
    model = Model()
    subdim = dims[j]
    XR = @variable(model, [1:subdim, 1:subdim], Symmetric)
    XI = @variable(model, [1:subdim, 1:subdim] in SkewSymmetricMatrixSpace())
    @constraint(model, tr(XR) == 1.0)
    @constraint(model, [XR XI; -XI XR] in PSDCone())

    # Construct the Kronecker product
    kron_prod = 1.0 + 0 * im
    for l in 1:nsubs
        if l == j
            kron_prod = @expression(model, kron(kron_prod, XR + im * XI))
        else
            principle = projectDensityMat(Xvals[:RE][l] + im * Xvals[:IM][l])
            kron_prod = @expression(model, kron(kron_prod, principle))
        end
    end

    sepaobj = @expression(model, dot(H[:RE], real(kron_prod)) + dot(H[:IM], imag(kron_prod)) )

    @objective(model, Max, sepaobj )

    Xs = Dict(:RE=>XR, :IM=>XI)
    optmodel = OptModel(model, Xs, nothing, nothing, nothing)
    return optmodel
end

function initalDensityMats(dims, seed)
    Vvals = [ rand(seed, dim, dim) + im * rand(seed, dim, dim)  for dim in dims]
    # Scale up the random matrices to avoid small values
    PSDs = [ 1000.0 * (V' * V) + 0.001 * Matrix{Complex}(I, dims[i], dims[i]) for (i,V) in enumerate(Vvals)]
    # Normalize to ensure trace is 1
    PSDs = [ PSD / tr(PSD) for PSD in PSDs]
    Xvals = Dict(:RE => [real(PSD) for PSD in PSDs], :IM => [imag(PSD) for PSD in PSDs])
    return Xvals
end

function AlternativeDescentCore(H, Xvals_, dims, seed, param)
    maxiter = 100
    if isnothing(Xvals_)
        Xvals = initalDensityMats(dims, seed)
    else
        Xvals = deepcopy(Xvals_)
    end
    nsubs = length(dims)
    bestHbar = constructFullSol(Xvals)
    bestobj = dot(real(bestHbar), H[:RE]) + dot(imag(bestHbar), H[:IM])
    bestXvals = Xvals
    prevobj = bestobj
    fail = 0
    for i in 1:maxiter
        sys = i % nsubs + 1
        optmodel = restrict(H, Xvals, nsubs, dims, sys)
        status, solverstatus, sol = solveModel(optmodel, nothing, param, true, true)
        if status == RelaxFeasible || status == RelaxOptimal
            Xval = projectDensityMat(sol.Xval[:RE] + im * sol.Xval[:IM])
            Xvals[:RE][sys] = real(Xval)
            Xvals[:IM][sys] = imag(Xval)
            Hbar = constructFullSol(Xvals)
            obj = dot(real(Hbar), H[:RE]) + dot(imag(Hbar), H[:IM])
            #if abs(obj - sol.primalobj) > 1e-3
            #    print("\n obj misatch", abs(obj - sol.primalobj))
            #end
            #print((i,sys))
            #validate(Xvals)
            if obj > bestobj
                bestXvals = deepcopy(Xvals)
                bestHbar = deepcopy(Hbar)
                bestobj = obj
            else
                fail += 1
            end
            if abs(obj - prevobj) < param.obj_tol
                break
            end
            if fail >= 5
                break
            end
            prevobj = obj
        else
            break
        end
        i += 1
    end
    return bestobj, bestXvals, bestHbar
end

function AlternativeDescent(stateseparator::StateSeparator, Xvals)
    problem = stateseparator.problem
    param = stateseparator.param
    dims = problem.dims
    bestobj, bestXvals, bestHbar = AlternativeDescentCore(problem.H, Xvals, dims, stateseparator.seed, param)
    #print("find a primal bound $(bestobj) by Alternative descent\n")
    if bestobj > stateseparator.primalbd
        stateseparator.primalbd = bestobj
        stateseparator.primalsol = bestXvals
        stateseparator.primalHbar = bestHbar
        stateseparator.primaloutbd = dot(problem.Hout[:RE], real(bestHbar)) + dot(problem.Hout[:IM], imag(bestHbar))
        if stateseparator.param.log_level > 0
            print("--find a better primal bound $(bestobj) by Alternative descent\n")
        end
        validate(bestXvals)
    end
    return bestXvals
end


function AlternativeDescentEigen(stateseparator::StateSeparator, Xvals_)
    problem = stateseparator.problem
    dimH = problem.dimH
    param = stateseparator.param
    dims = problem.dims
    nsubs = problem.nsubs
    maxiter = 100
    if isnothing(Xvals_)
        Xvals = initalDensityMats(dims, stateseparator.seed)
    else
        Xvals = deepcopy(Xvals_)
    end
    bestHbar = constructFullSol(Xvals)
    bestobj = dot(real(bestHbar), problem.H[:RE]) + dot(imag(bestHbar), problem.H[:IM])
    bestXvals = Xvals
    prevobj = bestobj
    fail = 0

    rH = tensorProductResahpe(problem.H[:RE] + im * problem.H[:IM], dims, nsubs)
    for i in 1:maxiter
        sys = i % nsubs + 1
        X = [Xvals[:RE][j] + im * Xvals[:IM][j] for j in 1:nsubs]
        gX = gfi(X, rH, dims, nsubs, sys)

        dim = dims[sys]
        rgX = [real(gX)' zeros(dim, dim); -imag(gX)' zeros(dim,dim)]
        rgX = (rgX + rgX') / 2
        eigvals, eigvecs = eigen(rgX)
        # Extract the eigenvector corresponding to the maximum eigenvalue
        maxndex = argmax(real(eigvals))   # Index of the maximum eigenvalue
        x = eigvecs[:, maxndex]     # Corresponding eigenvector

        # Construct xxᵀ
        principle = x * x'
        principle = principle * 2 / tr(principle)
        Xvals[:RE][sys] = principle[1:dim, 1:dim]
        Xvals[:IM][sys] = principle[1:dim, dim + 1 : 2 * dim]

        print((i,sys,tr(principle), tr(Xvals[:RE][sys]),  tr(Xvals[:RE][sys] + im * Xvals[:IM][sys])))
        validate(Xvals)
        Hbar = constructFullSol(Xvals)
        obj = dot(real(Hbar), problem.H[:RE]) + dot(imag(Hbar), problem.H[:IM])
        #if abs(obj - sol.primalobj) > 1e-3
        #    print("\n obj misatch", abs(obj - sol.primalobj))
        #end
        #print((i,sys))
        #validate(Xvals)
        if obj > bestobj
            bestXvals = deepcopy(Xvals)
            bestHbar = deepcopy(Hbar)
            bestobj = obj
        else
            fail += 1
        end
        if abs(obj - prevobj) < param.obj_tol
            break
        end
        if fail >= 5
            break
        end
        prevobj = obj
        i += 1
    end
    #print("find a primal bound $(bestobj) by Alternative descent\n")
    if bestobj > stateseparator.primalbd
        stateseparator.primalbd = bestobj
        stateseparator.primalsol = bestXvals
        stateseparator.primalHbar = bestHbar
        stateseparator.primaloutbd = dot(problem.Hout[:RE], real(bestHbar)) + dot(problem.Hout[:IM], imag(bestHbar))
        if stateseparator.param.log_level > 0
            print("--find a better primal bound $(bestobj) by Alternative descent Eigen\n")
        end
        validate(bestXvals)
    end
    return bestXvals
end

function ManifoldOpt(stateseparator::StateSeparator, H, Xvals_)
    problem = stateseparator.problem
    dims = problem.dims
    sol, _ = ManOptSolve(problem.dims, problem.nsubs, H, problem.dimH, Xvals_, stateseparator.seed, stateseparator.param)
    Hbar = constructFullSol(sol)
    primal = dot(real(Hbar), problem.H[:RE]) + dot(imag(Hbar), problem.H[:IM])
    #if !isnothing(stateseparator.primalsol)
    #    print("find primal bound $(primal) $(distance(Xvals_, sol)) $(distance(Xvals_, stateseparator.primalsol))\n")
    #end
    #print("find a primal bound $(primal) $(tr(Hbar)) $(tr(H)) by manifoldopt\n")
    if primal > stateseparator.primalbd
        if stateseparator.param.log_level > 0
            print("--find a better primal bound $(primal) by manifoldopt\n")
        end
        stateseparator.primalbd = primal
        stateseparator.primalsol = sol
        stateseparator.primalHbar = Hbar
        stateseparator.primaloutbd = dot(problem.Hout[:RE], real(Hbar)) + dot(problem.Hout[:IM], imag(Hbar))
    end
    return sol
end

function RunHeuristicsRoot(stateseparator::StateSeparator)
    problem = stateseparator.problem
    H = problem.H[:RE] +  im * problem.H[:IM]
    #FWHeuristic(stateseparator, H)
    #AlternativeDescentEigen(stateseparator::StateSeparator, nothing)
    AlternativeDescent(stateseparator, nothing)
    #Hbar_ = constructFullSol(stateseparator.primalsol)
    #primal_ = dot(real(Hbar_), problem.H[:RE]) + dot(imag(Hbar_), problem.H[:IM])
    #print("$(primal_) old obj \n")
    ManifoldOpt(stateseparator, H, stateseparator.primalsol)
end

function RunHeuristics(stateseparator::StateSeparator, sol)
    problem = stateseparator.problem
    H = problem.H[:RE] +  im * problem.H[:IM]
    #trivialHeuristic(stateseparator, H, sol.Xvals)
    if stateseparator.nodes[stateseparator.selectnode].depth % stateseparator.param.heur_AD_depth == 0
        #Xvals = AlternativeDescentEigen(stateseparator::StateSeparator, sol.Xvals)
        Xvals = AlternativeDescent(stateseparator::StateSeparator, sol.Xvals)
        ManifoldOpt(stateseparator, H, Xvals)
    end
    Xvals = ManifoldOpt(stateseparator::StateSeparator, H, sol.Xvals)
    return Xvals
end