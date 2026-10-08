using Test, LinearAlgebra, Random, Zygote
const E = ExactEntanglement

@testset "Lift (Psi)" begin
    dims = [2, 2, 2]; nsubs = length(dims); dimH = prod(dims)
    sumdim = sum(dims); cdims = E.cumulativeAdd!(copy(dims))

    rng = MersenneTwister(99)
    nr = 6
    subs = [[(v = randn(rng, ComplexF64, d); v ./ norm(v)) for d in dims] for _ in 1:nr]
    w = abs.(randn(rng, nr)); w ./= sum(w)

    p, r = E.packFactors(subs, w, nr, sumdim, dims, cdims, nsubs)
    M = E.LiftModel(r, dims, nsubs)

    @testset "pack/unpack round trip" begin
        @test r == nr
        @test length(p) == 2 * sumdim * nr
        @test eltype(p) == Float64
        @test E.manifold_dimension(M) == 2 * sumdim * nr - 1
        @test E.representation_size(M) == (length(p),)
        # weights fold into the vectors so that Psi(p) == sum_j w_j * p_j
        y = E.liftMap(M, p)
        ref = sum(w[j] * foldl(kron, [s * s' for s in subs[j]]) for j in 1:nr)
        @test y ≈ ref
        @test E.liftMap(M, p, 1) ≈ w[1] * foldl(kron, [s * s' for s in subs[1]])
        @test E.liftMap(M, p, 0) == 0
        @test E.fastTrace(M, p) ≈ 1
        @test real(tr(y)) ≈ 1

        _, _, w2 = E.unpackFactors(p, r, sumdim, dims, cdims, nsubs)
        @test sort(w2) ≈ sort(w)

        # Local scale freedom must not leak into CP factors. Include a zero
        # component and unequal local norms, preserving the original tensor.
        q = copy(p)
        q[1:4] .*= 10
        q[5:8] ./= 10
        q = vcat(q, zeros(2sumdim))
        P,S,W = E.unpackFactors(q,r+1,sumdim,dims,cdims,nsubs)
        @test length(W) == r
        @test all(norm(v) ≈ 1 for s in S for v in s)
        @test sum(W[j]*P[j] for j in eachindex(W)) ≈ y
        @test all(P[j] ≈ foldl(kron,[v*v' for v in S[j]]) for j in eachindex(W))
    end

    @testset "retraction keeps unit trace" begin
        g = randn(MersenneTwister(3), length(p))
        q = similar(p)
        E.retract_project!(M, q, p, 0.01 .* g)
        @test E.fastTrace(M, q) ≈ 1

        X = similar(g); E.fastProjTangent!(M,X,p,g)
        E.retract_project!(M,q,p,0.1X)
        Y = similar(X)
        E.ManifoldsBase.vector_transport_to!(M,Y,p,X,q,E.ProjectionTransport())
        normal = Zygote.gradient(x -> real(tr(E.liftMap(M,x))),q)[1]
        @test abs(dot(normal,Y)) <= 1e-12 * norm(normal) * norm(Y)
        # Transport really changes a tangent vector when the normal changes.
        @test abs(dot(normal,X)) > 1e-5
        @test E.fastTrace(M,q) ≈ 1
        # In-place changes must invalidate the cached trace normal.
        q .+= 0.02
        E.fastProjTangent!(M,Y,q,X)
        normal = Zygote.gradient(x -> real(tr(E.liftMap(M,x))),q)[1]
        @test abs(dot(normal,Y)) <= 1e-12 * norm(normal) * norm(Y)
    end

    @testset "analytic gradient matches AD" begin
        h = randn(MersenneTwister(5), ComplexF64, dimH, dimH); h = (h + h') / 2; h ./= tr(h)
        Min = ComplexF64.(Matrix(Diagonal(ones(dimH))) / dimH)
        Mdir = h - Min
        chi = fill(0.1 + 0.2im, dimH, dimH)
        al, fastgrad, _, _ = E.makeObjectiveClosures(M, Mdir, Min, chi, 1.7, 0.3, E.buildIndexMap(dims))

        analytic = fastgrad(M, p)
        ad = Zygote.gradient(x -> al(M, x), p)[1]
        projected = similar(ad); E.fastProjTangent!(M, projected, p, ad)
        # this is the core correctness invariant of LADMM
        @test analytic ≈ projected atol = 1e-10
        # The stationarity check must use the updated multiplier matrix.
        _, _, _, grad_l = E.makeObjectiveClosures(M, Mdir, Min, chi, 1.7, 0.3, nothing)
        chi .*= 2
        ad = Zygote.gradient(x -> real(dot(chi, E.liftMap(M, x))), p)[1]
        E.fastProjTangent!(M, projected, p, ad)
        @test grad_l(M, p) ≈ projected atol=1e-10

        cached_al,cached_grad,cached_func,cached_l = E.makeObjectiveClosures(
            M,Mdir,Min,chi,1.7,0.3,nothing; cached=true)
        for q in (copy(p),p .+ 0.01,copy(p))
            @test cached_al(M,q) ≈ al(M,q)
            @test cached_grad(M,q) ≈ fastgrad(M,q) atol=1e-10
            @test cached_func(M,q)[4] ≈ E.liftMap(M,q)-(0.3Mdir+Min)
            cached_l(M,q .+ 0.02)
            @test cached_grad(M,q) ≈ fastgrad(M,q) atol=1e-10
        end
    end
end

@testset "LADMM controls" begin
    limits = Param(heur_LADMM1_maxiter=8,heur_LADMM_maxiter=4,
        heur_MANOPT1_maxiter=200,heur_MANOPT_maxiter=150)
    @test E.ladmmIterationLimits(limits,true,false,32) == (8,200)
    @test E.ladmmIterationLimits(limits,false,false,32) == (4,150)
    @test E.ladmmPenalty(5.0, 1.0, 0.01, 0.1, :balance) == 10.0
    @test E.ladmmPenalty(5.0, 0.01, 1.0, 0.1, :balance) == 2.5
    @test E.ladmmPenalty(5.0, 0.1, 0.2, 0.1, :balance) == 5.0
    @test E.ladmmPenalty(200.0, 1.0, 0.0, 0.1, :balance) == 200.0
    @test E.ladmmPenalty(0.1, 0.0, 1.0, 0.1, :balance) == 0.1
    @test E.ladmmPenalty(5.0, 1.0, 0.01, 0.1, :legacy) == 12.5
    @test_throws ArgumentError Param(heur_LADMM_penalty_update=:invalid)

    # An exhausted budget returns the initial decomposition with its actual
    # residual, rather than falsely reporting feasibility without a solve.
    dims = [2, 2, 2]
    subs = [[ComplexF64[1, 0] for _ in dims]]
    H = Matrix{ComplexF64}(I, 8, 8) / 8
    chi = Dict(:RE=>zeros(8, 8), :IM=>zeros(8, 8))
    M = E.LiftModel(1, dims, 3)
    p, _ = E.packFactors(subs, [1.0], 1, sum(dims), dims, M.cdims, 3)
    linesearch = E.ladmmLineSearch(M, p)(M)
    @test linesearch.max_stepsize == 0.1
    @test linesearch.sufficient_curvature == 0.999
    @test linesearch.stop_when_stepsize_less > 0
    param = Param(time_limit=1.0, start_time=time()-2, log_level=0)
    pure, _, ub, residual, weights = E.ladmmSolve(nothing, dims, H, subs, [1.0], 0.5, chi, param)
    @test ub == 0.5
    @test residual ≈ norm(pure[1] - H)
    @test residual > 0.5
    @test sum(weights) ≈ 1.0

    # Exercise the configured line search and inner deadline in an actual solve.
    param = Param(time_limit=60.0, heur_LADMM1_maxiter=1,
        heur_MANOPT_maxiter=2, log_level=0)
    pure, _, _, residual, weights = E.ladmmSolve(nothing, dims, H, subs, [1.0], 0.5, chi, param)
    @test isfinite(residual)
    @test real(tr(sum(weights[i] * pure[i] for i in eachindex(weights)))) ≈ 1.0
end

@testset "LADMM preserves useful intermediate CP columns" begin
    dims = [2, 2]
    zero = ComplexF64[1, 0]; one = ComplexF64[0, 1]
    initial = [[zero, zero], [one, one]]
    final = reverse(initial)
    M = E.LiftModel(2, dims, 2)
    p, _ = E.packFactors(initial, [0.01, 0.99], 2, M.sumdim, dims, M.cdims, 2)
    q, _ = E.packFactors(final, [0.8, 0.2], 2, M.sumdim, dims, M.cdims, 2)
    witness = Matrix(Diagonal(ComplexF64[1, 0, 0, 0]))
    work = E.LiftGradientWorkspace(M)
    @test E.ladmmColumnScores(M, work, p, witness) ≈ [1, 0]
    @test E.ladmmColumnScores(M, work, q, witness) ≈ [0, 1]
    history = E.LADMMColumnHistory(copy(p), fill(-Inf, 2), witness)
    E.recordLADMMColumns!(history, M, work, p)
    E.recordLADMMColumns!(history, M, work, q)
    saved = copy(history.point)
    p .= 0   # stored states must survive mutation of Manopt's point array
    @test history.point == saved
    pool = (pure=Any[], sub=Any[])
    E.appendLADMMColumns!(pool, history, M, work, q, 0.5, 1e-6)
    @test length(pool.pure) == length(pool.sub) == 1
    @test real(dot(witness, pool.pure[1])) > 0.5
    @test pool.pure[1] ≈ foldl(kron, [v*v' for v in pool.sub[1]])
    @test all(norm(v) ≈ 1 for v in pool.sub[1])
    @test real(tr(pool.pure[1])) ≈ 1
    no_violation = (pure=Any[], sub=Any[])
    E.appendLADMMColumns!(no_violation, history, M, work, q, 1.0, 1e-6)
    @test isempty(no_violation.pure)
end

@testset "IR data correspondence" begin
    dims = [2, 2]
    rng = MersenneTwister(121)
    sub = [[normalize(randn(rng, ComplexF64, d)) for d in dims] for _ in 1:4]
    pure = [foldl(kron, [v*v' for v in factors]) for factors in sub]
    activeP, activeS, activeW = E.activeFactors(pure, sub, [0.0, 0.2, 0.0, 0.8])
    H = 0.2pure[2]+0.8pure[4]
    chi = Dict(:RE=>Matrix{Float64}(I,4,4), :IM=>zeros(4,4))
    w,s,z,_ = E.irWarmStart(activeP,activeS,activeW,4,H,0.0,chi,Param())
    @test w ≈ [0.8,0.2]
    @test s[1] == sub[4] && s[2] == sub[2]
    @test sum(w[a]*foldl(kron,[v*v' for v in s[a]]) for a in eachindex(w)) ≈ H
    @test_throws DimensionMismatch E.selectTopFactors(activeW,4,sub)
    p = Param(ir_refit_scalar=true)
    w,s,z,_ = E.irWarmStart(activeP,activeS,activeW,1,H,0.25,chi,p)
    mixed = Matrix{ComplexF64}(I,4,4)/4; B=H-mixed
    Y=foldl(kron,[v*v' for v in s[1]])
    @test norm(Y-(mixed+z*B)) <= norm(Y-(mixed+0.75B))+1e-12

    for (points,rank) in ((6,6),(6,3),(100,100))
        detector,w,tailP,tailS=E.initialActiveSet(real(H),imag(H),dims,
            Param(pointsize_bound=points,rank_bound=rank))
        @test length(tailP)==length(tailS)==max(length(w)-rank,0)
    end

    detector = E.ThresholdEntanglementDetector(real(H),imag(H),dims,pure,sub)
    expired = Param(time_limit=1.0,start_time=time()-2,log_level=0)
    ub,lb,_,P,S,w,_,total = E.cuttingPlane(detector,E.Problem(real(H),imag(H),dims),expired)
    @test ub==1 && lb==0 && total==1
    @test norm(sum(w[a]*P[a] for a in eachindex(w))-mixed)<1e-12
    @test all(P[a]≈foldl(kron,[v*v' for v in S[a]]) for a in eachindex(w))

    # A later failed/expired crossover must retain the preceding decomposition,
    # instead of reseeding LADMM with the computational basis and upper bound 1.
    old = E.MasterSnapshot(0.0,0.0,0.0,activeP,activeS,activeW,chi,1.0,0.0)
    saved = Ref{Union{Nothing,E.MasterSnapshot}}(old)
    ub,lb,_,P,S,w,multipliers,_ = E.cuttingPlane(detector,E.Problem(real(H),imag(H),dims),
        expired; lower_bound=0.0,snapshot_state=saved)
    @test ub == 0.0 && saved[] === old
    @test sum(w[a]*P[a] for a in eachindex(w)) ≈ H
    @test S == activeS
    E.flipMultipliers!(multipliers)
    @test saved[].witness[:RE] == chi[:RE]

    # An unchanged snapshot means that CP could only return the old
    # certificate. Continue the latest lifted factors without treating their
    # heuristic objective as a certified bound.
    cp_seed = (old.purestates, old.substates, old.weights, old.upper)
    lifted_seed = (pure, sub, fill(0.25,4), 0.3)
    @test E.irNextSeed(old, old, cp_seed, lifted_seed) === lifted_seed
    # A valid master solve creates independent arrays even when its upper
    # bound happens to be unchanged; mirror that snapshot construction here.
    newer = E.MasterSnapshot(old.upper,old.objective,old.offset,copy(old.purestates),
        copy(old.substates),copy(old.weights),old.witness,old.weights_sum,old.residual)
    P,S,w,u = E.irNextSeed(newer, old, cp_seed, lifted_seed)
    @test P == cp_seed[1] && S == cp_seed[2]
    @test w == cp_seed[3] && u == old.upper
end

@testset "Lift contraction gradient" begin
    rng = MersenneTwister(108)
    for dims in ([2], [2, 3], [2, 2, 2, 2], [2, 2, 2, 2, 2])
        M = E.LiftModel(3, dims, length(dims))
        p = 0.5 .* randn(rng, prod(E.representation_size(M)))
        # Include zero entries: dividing by a factor would give a wrong gradient.
        p[1:2] .= 0
        C = randn(rng, ComplexF64, prod(dims), prod(dims))
        work = E.LiftGradientWorkspace(M)
        g = similar(p)
        E.liftGradient!(M, g, p, C, nothing, work)
        reference = Zygote.gradient(x -> real(dot(C, E.liftMap(M, x))), p)[1]
        @test g ≈ reference atol=1e-10 rtol=1e-10
        # Reusing scratch storage must not retain derivatives from the last point.
        p .+= 0.1
        E.liftGradient!(M, g, p, C, nothing, work)
        reference = Zygote.gradient(x -> real(dot(C, E.liftMap(M, x))), p)[1]
        @test g ≈ reference atol=1e-10 rtol=1e-10
    end
end

@testset "Real-target CP conjugate columns" begin
    dims = [2, 2]
    rng = MersenneTwister(17)
    subs = [normalize(randn(rng, ComplexF64, d)) for d in dims]
    P = foldl(kron, [v * v' for v in subs])
    real_subs = [ComplexF64[1, 0] for _ in dims]
    Q = foldl(kron, [v * v' for v in real_subs])
    detector = E.ThresholdEntanglementDetector(Matrix{Float64}(I, 4, 4)/4,
        zeros(4, 4), dims, [], [])
    E.addBatchStates(detector, [P, Q], [subs, real_subs])
    detector.round = 2
    @test E.addConjugateStates!(detector, true) == 1
    @test detector.purestates[3] ≈ conj.(P)
    reconstructed = foldl(kron, [v * v' for v in detector.substates[3]])
    @test reconstructed ≈ detector.purestates[3]
    @test (P + detector.purestates[3])/2 ≈ real.(P)
    @test length(detector.purestates) == length(detector.substates) == length(detector.ispersistent)
    @test detector.poolstats == [2]
    detector.persistentInds = [1, 2]
    E.clearStates(detector)
    @test detector.ispersistent == [true, true]
    @test length(detector.purestates) == length(detector.ispersistent)
end
