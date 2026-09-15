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

function initalDensityMats(dims, seed)
    Vvals = [ rand(seed, dim, dim) + im * rand(seed, dim, dim)  for dim in dims]
    # Scale up the random matrices to avoid small values
    PSDs = [ 1000.0 * (V' * V) + 0.001 * Matrix{Complex}(I, dims[i], dims[i]) for (i,V) in enumerate(Vvals)]
    # Normalize to ensure trace is 1
    PSDs = [ PSD / tr(PSD) for PSD in PSDs]
    Xvals = Dict(:RE => [real(PSD) for PSD in PSDs], :IM => [imag(PSD) for PSD in PSDs])
    return Xvals
end

function partialInnerProductMap(A, C, D)
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
        gX = partialInnerProductMap(Matrix(kron_prod_left), Matrix(kron_prod_right), problem.H[:RE] + im * problem.H[:IM])
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

function RunHeuristicsRoot(stateseparator::StateSeparator)
    problem = stateseparator.problem
    AlternativeDescentEigen(stateseparator, nothing)
end

function RunHeuristics(stateseparator::StateSeparator, sol)
    problem = stateseparator.problem
    Xvals = AlternativeDescentEigen(stateseparator::StateSeparator, sol.Xvals)
    return Xvals
end

