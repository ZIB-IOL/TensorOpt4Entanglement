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
    # Contract the other factors directly, without slicing a block of D for
    # every entry. The conjugation is the Hilbert--Schmidt inner product.
    m = size(A, 1)
    p = size(C, 1)
    n = size(D, 1) ÷ (m * p)
    result = zeros(ComplexF64, n, n)
    @inbounds for δ in 1:p, γ in 1:m, β in 1:p, α in 1:m
        coefficient = conj(A[α, γ] * C[β, δ])
        for j in 1:n, i in 1:n
            row = (α - 1) * n * p + (i - 1) * p + β
            col = (γ - 1) * n * p + (j - 1) * p + δ
            result[i, j] += coefficient * D[row, col]
        end
    end
    return result
end

function AlternativeDescentEigen(stateseparator::StateSeparator, Xvals_)
    problem = stateseparator.problem
    param = stateseparator.param
    dims = problem.dims
    nsubs = problem.nsubs
    maxiter = param.heur_sbb_maxiter
    if isnothing(Xvals_)
        Xvals = initalDensityMats(dims, stateseparator.seed)
    else
        Xvals = deepcopy(Xvals_)
    end

    Xvals = principleSubStatesMat(Xvals)
    bestHbar = constructFullSol(Xvals)
    bestobj = dot(real(bestHbar), problem.H[:RE]) + dot(imag(bestHbar), problem.H[:IM])
    bestXvals = deepcopy(Xvals)
    sweepobj = bestobj
    H = problem.H[:RE] + im * problem.H[:IM]
    X = [Xvals[:RE][j] + im * Xvals[:IM][j] for j in 1:nsubs]

    for i in 1:maxiter
        remainingTime(param) <= 0 && break
        sys = i % nsubs + 1

        kron_prod_left = Matrix{ComplexF64}(I, 1, 1)
        kron_prod_right = Matrix{ComplexF64}(I, 1, 1)
        for l in 1:nsubs
            if l < sys
                kron_prod_left = kron(kron_prod_left, X[l])
            elseif l > sys
                kron_prod_right = kron(kron_prod_right, X[l])
            end
        end
        gX = partialInnerProductMap(kron_prod_left, kron_prod_right, H)
        eigvals, eigvecs = eigen(Hermitian((gX + gX') / 2))
        # Extract the eigenvector corresponding to the maximum eigenvalue
        maxndex = argmax(real(eigvals))   # Index of the maximum eigenvalue
        x = eigvecs[:, maxndex]     # Corresponding eigenvector

        # Construct xxᵀ
        principle = x * x'
        principle = principle / tr(principle)

        Xvals[:RE][sys] = real(principle)
        Xvals[:IM][sys] = imag(principle)
        X[sys] = principle

        Hbar = constructFullSol(Xvals)
        obj = dot(real(Hbar), problem.H[:RE]) + dot(imag(Hbar), problem.H[:IM])

        if obj > bestobj
            bestXvals = deepcopy(Xvals)
            bestHbar = deepcopy(Hbar)
            bestobj = obj
        end
        # One stationary coordinate does not imply stationarity of the other
        # factors. Test convergence only after all subsystems have been visited.
        if i % nsubs == 0
            bestobj - sweepobj < param.obj_tol && break
            sweepobj = bestobj
        end
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
    result = nothing
    for _ in 1:stateseparator.param.heur_sbb_restarts
        remainingTime(stateseparator.param) <= 0 && break
        result = AlternativeDescentEigen(stateseparator, nothing)
    end
    return result
end

function RunHeuristics(stateseparator::StateSeparator, sol)
    bestX = AlternativeDescentEigen(stateseparator, sol.Xvals)
    starts = stateseparator.param.heur_sbb_node_restarts
    (starts == 1 || remainingTime(stateseparator.param) <= 0) && return bestX

    # Sample pure factors with covariance given by the relaxed marginal. A
    # small isotropic component also explores directions omitted by nearly
    # rank-one marginals. Every sample is still a feasible product state.
    transforms = Matrix{ComplexF64}[]
    for i in eachindex(sol.Xvals[:RE])
        X = sol.Xvals[:RE][i] + im * sol.Xvals[:IM][i]
        λ,U = eigen(Hermitian((X + X') / 2))
        λ = max.(λ,0.0)
        total = sum(λ)
        λ = total > 0 ? λ / total : fill(1 / length(λ),length(λ))
        push!(transforms,U * Diagonal(sqrt.(0.95 .* λ .+ 0.05 / length(λ))))
    end
    H = stateseparator.problem.H[:RE] + im * stateseparator.problem.H[:IM]
    bestscore = real(dot(H,constructFullSol(bestX)))
    for _ in 2:starts
        remainingTime(stateseparator.param) <= 0 && break
        factors = [T * randn(stateseparator.seed,ComplexF64,size(T,2)) for T in transforms]
        densities = [v * v' / real(dot(v,v)) for v in factors]
        initial = Dict(:RE=>real.(densities),:IM=>imag.(densities))
        X = AlternativeDescentEigen(stateseparator,initial)
        score = real(dot(H,constructFullSol(X)))
        if score > bestscore
            bestX,bestscore = X,score
        end
    end
    return bestX
end
