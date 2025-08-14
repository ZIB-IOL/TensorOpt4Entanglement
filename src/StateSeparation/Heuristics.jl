using TensorOperations
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


function principleSubStates(Xvals)
    nsubs = length(Xvals[:RE])
    substates = []
    for i in 1:nsubs
        X = Xvals[:RE][i] + im * Xvals[:IM][i]
        eigvals, eigvecs = eigen(X)
        # Extract the eigenvector corresponding to the maximum eigenvalue
        maxndex = argmax(real(eigvals))   # Index of the maximum eigenvalue
        x = eigvecs[:, maxndex]     # Corresponding eigenvector
        # normalize the eigenvector
        x = x / norm(x)
        push!(substates, x)
    end
    return substates
end

function principleSubStatesMat(Xvals)
    nsubs = length(Xvals[:RE])
    Xvals_ = Dict(:RE=>[], :IM=>[])
    for i in 1:nsubs
        X = Xvals[:RE][i] + im * Xvals[:IM][i]
        X = projectDensityMat(X)
        push!(Xvals_[:RE], real(X))
        push!(Xvals_[:IM], imag(X))
    end
    return Xvals_
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
            principle = Xvals[:RE][l] + im * Xvals[:IM][l]
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


function AlternativeDescentSDP(H, Xvals_, dims, seed, param)
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

    Xvals = principleSubStatesMat(Xvals)
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

function partialfunc(X, kron_prod_left, kron_prod_right, H)
    kron_prod = kron(kron_prod_left, X , kron_prod_right)
    return dot(H[:RE], real(kron_prod)) + dot(H[:IM], imag(kron_prod))
end

function compute_inner_product_map_vectorized(A, C, D)
    """
    Most efficient vectorized version using Einstein summation pattern.

    We use the fact that:
    ⟨A ⊗ E_ij ⊗ C, D⟩ = Σ_{α,β,γ,δ} A[α,γ] * C[β,δ] * D[row,col]
    where row = (α-1)*n*p + (i-1)*p + β + 1
          col = (γ-1)*n*p + (j-1)*p + δ + 1
    """
    m, _ = size(A)
    p, _ = size(C)
    total_size = size(D, 1)
    n = total_size ÷ (m * p)

    result = zeros(ComplexF64, n, n)

    # Pre-compute all possible row and column indices
    for i in 1:n, j in 1:n
        # Use broadcasting to compute all combinations at once
        α_range = 1:m
        β_range = 1:p
        γ_range = 1:m
        δ_range = 1:p

        # Compute row and column indices for all combinations
        rows = [(α-1)*n*p + (i-1)*p + β for α in α_range, β in β_range]
        cols = [(γ-1)*n*p + (j-1)*p + δ for γ in γ_range, δ in δ_range]

        # Extract the relevant submatrix from D
        D_sub = D[rows[:], cols[:]]  # m*p × m*p matrix

        # Reshape to separate the A and C components
        D_reshaped = reshape(D_sub, m, p, m, p)

        # Contract with A and C using Einstein summation
        # result[i,j] = Σ_{α,β,γ,δ} A[α,γ] * C[β,δ] * D[α,β,γ,δ]
        result[i, j] = sum( real(A[α,γ] * C[β,δ]) * real(D_reshaped[α,β,γ,δ]) + imag(A[α,γ] * C[β,δ]) * imag(D_reshaped[α,β,γ,δ])
            + im * (real(A[α,γ] * C[β,δ]) * imag(D_reshaped[α,β,γ,δ]) - imag(A[α,γ] * C[β,δ]) * real(D_reshaped[α,β,γ,δ]))
            for α in 1:m, β in 1:p, γ in 1:m, δ in 1:p)
    end

    return result
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

    Xvals = principleSubStatesMat(Xvals)
    bestHbar = constructFullSol(Xvals)
    bestobj = dot(real(bestHbar), problem.H[:RE]) + dot(imag(bestHbar), problem.H[:IM])
    bestXvals = Xvals
    prevobj = bestobj
    fail = 0

    for i in 1:maxiter
        sys = i % nsubs + 1
        X = [Xvals[:RE][j] + im * Xvals[:IM][j] for j in 1:nsubs]

        kron_prod_left = Matrix{ComplexF64}(I, 1, 1)
        kron_prod_right = Matrix{ComplexF64}(I, 1, 1)
        for l in 1:nsubs
            if l < sys
                kron_prod_left = kron(kron_prod_left, X[l])
            elseif l > sys
                kron_prod_right = kron(kron_prod_right, X[l])
            end
        end
        #gX = gradient(p -> partialfunc(p, kron_prod_left, kron_prod_right, problem.H), X[sys])[1]
        gX = compute_inner_product_map_vectorized(Matrix(kron_prod_left), Matrix(kron_prod_right), problem.H[:RE] + im * problem.H[:IM])
        #gX = real(gX) + imag(gX) + im * ( real(gX) - imag(gX) )
        rgX = (gX + gX') / 2
        eigvals, eigvecs = eigen(rgX)
        # Extract the eigenvector corresponding to the maximum eigenvalue
        maxndex = argmax(real(eigvals))   # Index of the maximum eigenvalue
        x = eigvecs[:, maxndex]     # Corresponding eigenvector

        # Construct xxᵀ
        principle = x * x'
        principle = principle / tr(principle)

        Xvals[:RE][sys] = real(principle)
        Xvals[:IM][sys] = imag(principle)

        Hbar = constructFullSol(Xvals)
        obj = dot(real(Hbar), problem.H[:RE]) + dot(imag(Hbar), problem.H[:IM])

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

function SphereOpt(stateseparator::StateSeparator, H, Xvals_)
    problem = stateseparator.problem
    dims = problem.dims
    sol, _ = SphereSolve(problem.dims, problem.nsubs, H, problem.dimH, Xvals_, stateseparator.seed, stateseparator.param)
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

function AlternativeDescent(stateseparator::StateSeparator, Xvals)
    problem = stateseparator.problem
    param = stateseparator.param
    dims = problem.dims
    bestobj, bestXvals, bestHbar = AlternativeDescentSDP(problem.H, Xvals, dims, stateseparator.seed, param)
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


function RunHeuristicsRoot(stateseparator::StateSeparator)
    problem = stateseparator.problem
    #H = problem.H[:RE] +  im * problem.H[:IM]
    AlternativeDescentEigen(stateseparator, nothing)
    #SphereOpt(stateseparator, H, nothing)
end

function RunHeuristics(stateseparator::StateSeparator, sol)
    problem = stateseparator.problem
    #H = problem.H[:RE] +  im * problem.H[:IM]
    Xvals = AlternativeDescentEigen(stateseparator::StateSeparator, sol.Xvals)
    #Xvals = SphereOpt(stateseparator::StateSeparator, H, sol.Xvals)
    return Xvals
end