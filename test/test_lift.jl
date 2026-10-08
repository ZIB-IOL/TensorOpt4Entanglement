using Test, LinearAlgebra, Random, Zygote
const E = ExactEntanglement

@testset "CP termination uses the certified master upper bound" begin
    detector = E.ThresholdEntanglementDetector(zeros(4,4),zeros(4,4),[2,2],[],[])
    p = Param(log_level=0)
    upper = 0.8
    selected_objective = upper - 100p.master_obj_tol
    terminate,addstate,lower = E.checkTerminationGap(detector,selected_objective,
        0.2,0.2,0.2,0.0,p;master_upper=upper)
    @test !terminate && !addstate
    @test lower == selected_objective
    terminate,addstate,lower = E.checkTerminationGap(detector,upper,
        0.2,0.2,0.2,0.0,p;master_upper=upper)
    @test terminate && !addstate
    @test lower == upper
    previous_lower = upper - p.master_obj_tol/2
    terminate,addstate,lower = E.checkTerminationGap(detector,selected_objective,
        0.2,1.0,0.2,previous_lower,p;master_upper=upper)
    @test terminate && !addstate
    @test lower < previous_lower
end

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

        small = [1-5e-10, 5e-10]
        tiny_p, tiny_rank = E.packFactors(subs[1:2], small, 2, sumdim, dims, cdims, nsubs)
        @test tiny_rank == 2
        tiny_model = E.LiftModel(tiny_rank, dims, nsubs)
        @test E.fastTrace(tiny_model,tiny_p) ≈ 1 atol=1e-14
        @test E.liftMap(tiny_model,tiny_p) ≈
            sum(small[j]*foldl(kron,[v*v' for v in subs[j]]) for j in 1:2) atol=1e-14
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

    @testset "retraction recovers invalid tangent trials" begin
        model = E.LiftModel(1, dims, nsubs)
        base = zeros(2sumdim)
        base[[1,5,9]] = [0.01,0.01,10000.0]
        tangent = zeros(length(base))
        tangent[[1,5]] = [-0.01,0.01]
        projected = similar(tangent)
        E.fastProjTangent!(model,projected,base,tangent)
        @test projected ≈ tangent atol=1e-15
        @test norm(tangent) < E.max_stepsize(model)
        @test E.fastTrace(model,base) ≈ 1
        @test E.fastTrace(model,base+tangent) == 0

        for inplace in (false,true)
            input = copy(base)
            output = inplace ? input : similar(input)
            @test E.retract_project!(model,output,input,tangent) === output
            @test all(isfinite,output)
            @test output == base
            @test E.fastTrace(model,output) ≈ 1

            # The next smaller trial uses the original normalization formula.
            trial = base + 0.5tangent
            expected = trial / E.fastTrace(model,trial)^(1/(2nsubs))
            E.retract_project!(model,output,input,0.5tangent)
            @test output ≈ expected atol=1e-14
            @test output != base
            @test E.fastTrace(model,output) ≈ 1
        end

        invalid = fill(Inf,length(base))
        output = similar(base)
        E.retract_project!(model,output,base,invalid)
        @test output == base
        E.retract_project!(model,output,output,invalid)
        @test output == base
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

        # The updated Lagrangian gradient includes both inner-solve error
        # and the scalar block's movement, with the unscaled ADMM multiplier.
        inner_gradient = fastgrad(M,p)
        scalar_change = 0.1
        chi .+= 2 * 1.7 .* (E.liftMap(M,p)-(0.4Mdir+Min))
        direction_gradient = zeros(length(p))
        E.liftGradient!(M,direction_gradient,p,Mdir,nothing)
        E.fastProjTangent!(M,direction_gradient,p,direction_gradient)
        @test grad_l(M,p) ≈ inner_gradient-2 * 1.7 * scalar_change * direction_gradient atol=1e-10
    end
end

@testset "LADMM Wolfe search terminates at floating-point brackets" begin
    M = E.Manifolds.Euclidean(1)
    p = [0.0]
    cost = (M,p) -> p[1]
    gradient = (M,p) -> [1.0]
    problem = E.Manopt.DefaultManoptProblem(M,
        E.Manopt.ManifoldGradientObjective(cost,gradient))
    search = E.Manopt.WolfePowellLinesearchStepsize(M;p=copy(p),X=zeros(1),
        max_stepsize=0.1,stop_when_stepsize_less=1e-10)
    guard = E.LADMMWolfeGuard(search)
    state = E.Manopt.QuasiNewtonState(M;p=copy(p),X=gradient(M,p),stepsize=guard)
    @test search.stop_when_stepsize_less == 1e-10

    # A linear objective has constant directional derivative, so Wolfe's
    # curvature condition never holds. A tiny direction makes the scalar
    # bracket large enough that its adjacent floats exceed the old tolerance.
    direction = [-1e-8]
    cap = min(1e9,search.max_stepsize/norm(direction))
    @test cap == 1e7
    step = guard(problem,state,1,direction)
    @test isfinite(step) && 0 < step <= 2cap
    @test cost(M,p+step*direction) <=
        cost(M,p)+search.sufficient_decrease*step*dot(gradient(M,p),direction)
    @test search.stop_when_stepsize_less == max(1e-10,8eps(cap))
    @test E.Manopt.get_last_stepsize(guard) == step
    lo = 2cap; hi = nextfloat(lo)
    @test (lo+hi)/2 in (lo,hi)
    @test hi-lo > 1e-10
    @test hi-lo <= search.stop_when_stepsize_less

    # The safeguard must reset, rather than retain a large scalar tolerance.
    step = guard(problem,state,2,[-1.0])
    @test isfinite(step) && step > 0
    @test search.stop_when_stepsize_less == 1e-10
    @test E.Manopt.get_last_stepsize(guard) == step
    @test guard(problem,state,3,[0.0]) == 1.0
    @test search.stop_when_stepsize_less == max(1e-10,8eps(1e9))
    guard(problem,state,4,[-1.0])
    @test search.stop_when_stepsize_less == 1e-10
end

@testset "LADMM Wolfe search enforces descent and its deadline" begin
    M = E.Manifolds.Euclidean(1)
    search_for(p) = E.Manopt.WolfePowellLinesearchStepsize(M;p=copy(p),X=zeros(1),
        max_stepsize=0.1,stop_when_stepsize_less=1e-10)
    # With ||direction|| > 1e8 the initial scalar step lies below Wolfe's
    # absolute backtracking cutoff. Its returned trial can increase the cost
    # or lie outside a finite-cost domain; both need actual backtracking.
    for curvature in (1e11, 1e15), finite_domain in (false, true)
        p = [0.01]
        cost = (M,q) -> finite_domain && q[1] <= 0 ? NaN : curvature*q[1]^2/2
        gradient = (M,q) -> [curvature*q[1]]
        problem = E.Manopt.DefaultManoptProblem(M,
            E.Manopt.ManifoldGradientObjective(cost,gradient))
        search = search_for(p)
        guard = E.LADMMWolfeGuard(search)
        state = E.Manopt.QuasiNewtonState(M;p=copy(p),X=gradient(M,p),stepsize=guard)
        direction = -gradient(M,p)
        step = guard(problem,state,1,direction)
        trial = p + step*direction
        @test isfinite(step) && step > 0
        @test isfinite(cost(M,trial))
        @test cost(M,trial) < cost(M,p)
        @test cost(M,trial) <= cost(M,p) +
            search.sufficient_decrease*step*dot(gradient(M,p),direction)
        @test E.Manopt.get_last_stepsize(guard) == step
        @test state.p == p
    end

    # A normal Wolfe step remains unchanged, including a small direction.
    for p in ([1.0], [1e-8])
        cost = (M,q) -> q[1]^2/2
        gradient = (M,q) -> copy(q)
        problem = E.Manopt.DefaultManoptProblem(M,
            E.Manopt.ManifoldGradientObjective(cost,gradient))
        raw = search_for(p)
        guard = E.LADMMWolfeGuard(search_for(p),time()+120)
        state = E.Manopt.QuasiNewtonState(M;p=copy(p),X=gradient(M,p),stepsize=guard)
        direction = -gradient(M,p)
        expected = raw(problem,state,1,direction)
        @test guard(problem,state,1,direction) == expected
        @test E.Manopt.get_last_stepsize(guard) == expected
    end

    # An expired search returns no step, without evaluating the objective.
    p = [0.01]
    evaluations = Ref(0)
    cost = (M,q) -> (evaluations[] += 1; q[1]^2/2)
    gradient = (M,q) -> copy(q)
    problem = E.Manopt.DefaultManoptProblem(M,
        E.Manopt.ManifoldGradientObjective(cost,gradient))
    guard = E.LADMMWolfeGuard(search_for(p),time()-1)
    state = E.Manopt.QuasiNewtonState(M;p=copy(p),X=gradient(M,p),stepsize=guard)
    @test guard(problem,state,1,-p) == 0.0
    @test E.Manopt.get_last_stepsize(guard) == 0.0
    @test evaluations[] == 0
    @test state.p == p

    # Stop inside Wolfe when a cost call exhausts the budget, instead of
    # waiting for the whole bracket search. Preserve the accepted solver point.
    deadline = Ref(time()+120)
    wait_for_deadline = Ref(false)
    slow_cost = (M,q) -> begin
        evaluations[] += 1
        wait_for_deadline[] && evaluations[] == 2 &&
            sleep(max(0.0,deadline[]-time())+0.01)
        q[1]^2/2
    end
    problem = E.Manopt.DefaultManoptProblem(M,
        E.Manopt.ManifoldGradientObjective(slow_cost,gradient))
    # Compile this callback path before starting its short time budget.
    guard = E.LADMMWolfeGuard(search_for(p),deadline[])
    state = E.Manopt.QuasiNewtonState(M;p=copy(p),X=gradient(M,p),stepsize=guard)
    @test guard(problem,state,1,-p) > 0
    deadline[] = time()+0.5
    evaluations[] = 0
    wait_for_deadline[] = true
    guard = E.LADMMWolfeGuard(search_for(p),deadline[])
    @test guard(problem,state,1,-p) == 0.0
    @test E.Manopt.get_last_stepsize(guard) == 0.0
    @test evaluations[] == 2
    @test state.p == p
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

    requested = (1e-6, 1e-5, 1e-5)
    @test E.ladmmInnerTolerances(requested..., 0.0, false) == requested
    @test E.ladmmInnerTolerances(requested..., 1.0, true) == requested
    forcing = E.ladmmInnerTolerances(requested..., 1e-3, true)
    @test forcing[1] ≈ 5e-8
    @test forcing[2:3] == requested[2:3]
    tighter = E.ladmmInnerTolerances(requested..., 1e-6, true)
    @test all(tighter .<= forcing)
    @test tighter[2] ≈ 1e-7
    @test tighter[3] ≈ 1e-7
    # Positive floors avoid requesting exact stationarity and never weaken a
    # stricter user request.
    @test E.ladmmInnerTolerances(requested..., 0.0, true) == (1e-12, 1e-9, 1e-7)
    stricter = (1e-14, 1e-11, 1e-9)
    @test E.ladmmInnerTolerances(stricter..., 0.0, true) == stricter

    stopping = Param(feas_tol=1e-6, master_obj_tol=1e-6)
    stable = fill(0.8, 4)
    # Inner cost tolerances must not silently demand tighter outer feasibility.
    @test E.ladmmConverged(8e-7, 1e-7, 1e-6, stable, stopping)
    @test !E.ladmmConverged(2e-6, 1e-7, 1e-6, stable, stopping)
    @test !E.ladmmConverged(8e-7, 2e-6, 1e-6, stable, stopping)
    @test !E.ladmmConverged(8e-7, 1e-7, 1e-6, stable[1:3], stopping)
    # Several small objective changes can still accumulate appreciable progress.
    changing = 0.8 .+ (0:3) .* 5e-7
    @test !E.ladmmConverged(8e-7, 1e-7, 1e-6, changing, stopping)
    @test !E.ladmmConverged(8e-7, 1e-7, 1e-6, Float64[], stopping)
    @test !E.ladmmConverged(2e-6, 1e-7, 1e-5, stable, stopping)

    # A worse raw objective can be useful when the final infeasibility makes
    # that objective uncertain. A much worse objective or equal residual is
    # not retained, and the identity direction cannot supply a scalar band.
    @test E.ladmmKeepEarlierPoint(1e-4, 0.9001, 1e-3, 0.9, 1.0, 1e-6)
    @test !E.ladmmKeepEarlierPoint(1e-4, 0.91, 1e-3, 0.9, 1.0, 1e-6)
    @test !E.ladmmKeepEarlierPoint(1e-3, 0.9, 1e-3, 0.9, 1.0, 1e-6)
    @test !E.ladmmKeepEarlierPoint(1e-4, 0.9, 1e-3, 0.9, 0.0, 1e-6)

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

@testset "LADMM continues its dual state after a failed CP pass" begin
    dims = [2, 2]
    D = prod(dims)
    H = Matrix{ComplexF64}(I, D, D) / D
    subs = [[ComplexF64[1, 0] for _ in dims]]
    chi = Dict(:RE=>zeros(D, D), :IM=>zeros(D, D))
    param = Param(time_limit=60.0, log_level=0, heur_LADMM_rho=3.0,
        heur_LADMM_penalty_update=:balance, heur_LADMM1_maxiter=1,
        heur_LADMM_maxiter=1, heur_MANOPT1_maxiter=2, heur_MANOPT_maxiter=2)
    state = Ref{Any}(nothing)
    pure, sub, ub, _, weights = E.ladmmSolve(nothing, dims, H, subs, [1.0],
        0.5, chi, param; warm_state=state)
    residual = pure[1] - H
    first = state[]
    @test first.multipliers ≈ 6residual
    @test first.penalty == 6.0
    @test iszero(norm(chi[:RE])) && iszero(norm(chi[:IM]))

    # With the same primal iterate, restarting from the old CP multiplier
    # would repeat 6R. Continuation instead applies the next update to 6R,
    # using the carried penalty: 6R + 2*6R = 18R.
    E.ladmmSolve(nothing, dims, H, sub, weights, 1-ub, chi, param;
        warm_state=state)
    @test state[].multipliers ≈ 18residual
    @test state[].penalty == 12.0
    @test first.multipliers ≈ 6residual

    # Budget exhaustion must retain a copied state without an extra update.
    expired = Param(time_limit=1.0, start_time=time()-2, log_level=0)
    saved = state[]
    E.ladmmSolve(nothing, dims, H, sub, weights, 1-ub, chi, expired;
        warm_state=state)
    @test state[].multipliers == saved.multipliers
    @test state[].multipliers !== saved.multipliers
    @test state[].penalty == saved.penalty
    wrong = Ref{Any}((multipliers=zeros(ComplexF64, 2, 2), penalty=1.0))
    @test_throws DimensionMismatch E.ladmmSolve(nothing, dims, H, sub,
        weights, 1-ub, chi, expired; warm_state=wrong)
end

@testset "LADMM keeps large CP multipliers and scalar stationarity" begin
    dims = [2,2]; D = prod(dims)
    sigma = Matrix{ComplexF64}(I,D,D)/D
    epsilon = 1e-4
    H = copy(sigma)
    H[1,4] = H[4,1] = epsilon
    B = H-sigma
    sub = [[ComplexF64[j==index[k] for j in 1:d] for (k,d) in enumerate(dims)]
        for index in Iterators.product((1:d for d in dims)...)]
    sub = vec(sub)
    pure = [foldl(kron,[v*v' for v in s]) for s in sub]
    weights = fill(1.0/D,D)
    witness = zeros(ComplexF64,D,D)
    witness[1,4] = witness[4,1] = 1/(2epsilon)
    # This is a valid finite-master witness with offset zero and objective 1:
    # every computational product column has witness value zero.
    @test real(dot(witness,B)) ≈ 1
    @test all(iszero(real(dot(witness,P))) for P in pure)
    @test real(dot(witness,H)) ≈ 1
    @test minimum(eigvals(Hermitian(H))) > 0
    chi = Dict(:RE=>real(-witness),:IM=>imag(-witness))
    param = Param(time_limit=120.0,is_last=true,log_level=0,
        heur_LADMM1_maxiter=1,heur_LADMM_maxiter=1,
        heur_MANOPT1_maxiter=2,heur_MANOPT_maxiter=2,
        heur_LADMM_penalty_update=:balance)
    state = Ref{Any}(nothing)
    result = E.ladmmSolve(nothing,dims,H,sub,weights,0.0,chi,param;
        warm_state=state)
    @test state[].iterations == 1
    @test state[].z == 0.0
    @test maximum(abs,state[].multipliers) > 100
    @test state[].multipliers[1,4] ≈ -1/(2epsilon)
    @test real(dot(state[].multipliers,B)) ≈ -1 atol=1e-12
    # At z=0 the scalar KKT derivative must be nonnegative. The exact z-step
    # makes it zero here; clipping entries at 100 would instead give -0.98.
    scalar_gradient = -1-real(dot(state[].multipliers,B))
    @test abs(scalar_gradient) <= 1e-12
    clipped = complex.(clamp.(real(state[].multipliers),-100,100),
        clamp.(imag(state[].multipliers),-100,100))
    @test -1-real(dot(clipped,B)) < -0.9
    returned = sum(result[5][j]*result[1][j] for j in eachindex(result[5]))
    @test result[4] ≈ norm(returned-sigma) atol=1e-14
    @test result[4] <= 1e-12

    # The fixed point satisfies feasibility and both scalar/manifold KKT
    # conditions. Both modes check objective stability across the initial point
    # and three updates, then preserve that history.
    ordinary = deepcopy(param)
    ordinary.heur_LADMM_maxiter = ordinary.heur_LADMM1_maxiter = 5
    ordinary_state = Ref{Any}(nothing)
    E.ladmmSolve(nothing,dims,H,sub,weights,0.0,chi,ordinary;
        warm_state=ordinary_state)
    @test ordinary_state[].iterations == 3
    @test ordinary_state[].objectives ≈ ones(4)
    accurate = Ref{Any}(nothing)
    E.ladmmSolve(nothing,dims,H,sub,weights,0.0,chi,param,false,true;
        warm_state=accurate)
    @test accurate[].iterations == 3
    @test accurate[].objectives ≈ ones(4)
    saved_objectives = accurate[].objectives
    expired = Param(time_limit=1.0,start_time=time()-2,log_level=0)
    E.ladmmSolve(nothing,dims,H,sub,weights,0.0,chi,expired,false,true;
        warm_state=accurate)
    @test accurate[].iterations == 3
    @test accurate[].objectives == saved_objectives
    @test accurate[].objectives !== saved_objectives
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

    # A zero slot is omitted by unpackFactors. Original rank indices must
    # therefore be selected before unpacking, rather than indexing its output.
    zero_slot = zeros(2M.sumdim)
    live, _ = E.packFactors([[zero,zero]],[1.0],1,M.sumdim,dims,M.cdims,2)
    last, _ = E.packFactors([[one,one]],[1.0],1,M.sumdim,dims,M.cdims,2)
    zero_history = E.LADMMColumnHistory(vcat(zero_slot,live),[-Inf,1.0],witness)
    selected = (pure=Any[],sub=Any[])
    E.appendLADMMColumns!(selected,zero_history,M,work,vcat(zero_slot,last),0.5,1e-6)
    @test length(selected.pure) == length(selected.sub) == 1
    @test selected.pure[1] ≈ foldl(kron,[zero*zero',zero*zero'])
    @test selected.pure[1] ≈ foldl(kron,[v*v' for v in selected.sub[1]])
end

@testset "LADMM restores the complete lifted iterate on CP fallback" begin
    dims = [2,2]; D = prod(dims)
    bell = ComplexF64[1,0,0,1]/sqrt(2)
    H = bell*bell'
    rng = MersenneTwister(131)
    initial = [[normalize(randn(rng,ComplexF64,d)) for d in dims] for _ in 1:4]
    weights = fill(0.25,4)
    chi = Dict(:RE=>zeros(D,D),:IM=>zeros(D,D))
    function continuation_params(iterations)
        Param(time_limit=120.0,is_last=true,log_level=0,rank_bound=4,
            heur_LADMM1_maxiter=iterations,heur_LADMM_maxiter=iterations,
            heur_MANOPT1_maxiter=2,heur_MANOPT_maxiter=2,
            heur_LADMM_penalty_update=:balance,ir_refit_scalar=true)
    end
    mixture(result) = sum(result[5][j]*result[1][j] for j in eachindex(result[5]))
    uninterrupted_state = Ref{Any}(nothing)
    uninterrupted = E.ladmmSolve(nothing,dims,H,initial,weights,0.3,chi,
        continuation_params(2),true;warm_state=uninterrupted_state)
    continued_state = Ref{Any}(nothing)
    first = E.ladmmSolve(nothing,dims,H,initial,weights,0.3,chi,
        continuation_params(1),true;warm_state=continued_state)
    first_state = continued_state[]
    p = continuation_params(1)
    w,s,refitted_z,_ = E.irWarmStart(first[1],first[2],first[5],4,H,first[3],chi,p)
    @test abs(refitted_z-first_state.z) > 1e-3
    # Even an incoming scalar refit must not overwrite the saved fallback
    # iterate. Both calls use equal inner limits, so 1+1 reproduces 2 steps.
    resumed = E.ladmmSolve(nothing,dims,H,s,w,refitted_z,chi,p;
        warm_state=continued_state)
    @test mixture(resumed) ≈ mixture(uninterrupted) atol=1e-11 rtol=1e-11
    @test resumed[3] ≈ uninterrupted[3] atol=1e-12
    @test resumed[4] ≈ uninterrupted[4] atol=1e-12
    @test continued_state[].point ≈ uninterrupted_state[].point atol=1e-11
    @test continued_state[].multipliers ≈ uninterrupted_state[].multipliers atol=1e-11
    @test continued_state[].penalty == uninterrupted_state[].penalty
    @test continued_state[].tolerances == uninterrupted_state[].tolerances
    @test continued_state[].iterations == uninterrupted_state[].iterations == 2
    @test continued_state[].objectives ≈ uninterrupted_state[].objectives atol=1e-12
    @test length(continued_state[].objectives) == 3

    # Preserve nonuniform local scales and zero slots, neither of which
    # survives unpacking and repacking the normalised product factors.
    zero = ComplexF64[1,0]; one = ComplexF64[0,1]
    sub = [[zero,zero],[one,one]]
    M = E.LiftModel(2,dims,2)
    point,_ = E.packFactors(sub,[0.7,0.3],2,M.sumdim,dims,M.cdims,2)
    point[1:4] .*= 5
    point[5:8] ./= 5
    point = vcat(point,zeros(2M.sumdim))
    state = Ref{Any}((multipliers=zeros(ComplexF64,D,D),penalty=2.0,
        point=point,z=0.7,dims=copy(dims),tolerances=(1e-5,1e-4,1e-4),iterations=5))
    expired = Param(time_limit=1.0,start_time=time()-2,log_level=0)
    result = E.ladmmSolve(nothing,dims,H,sub,[0.7,0.3],0.1,chi,expired;warm_state=state)
    @test result[3] ≈ 0.3
    @test state[].point ≈ point atol=1e-13
    @test length(state[].point) == 3*2M.sumdim
    @test length(result[5]) == 2
    @test state[].iterations == 5
    @test state[].point !== point
    @test state[].dims !== dims
    sigma = Matrix{ComplexF64}(I,D,D)/D
    @test result[4] ≈ norm(mixture(result)-(sigma+0.7*(H-sigma))) atol=1e-14

    # Older multiplier/penalty-only packets remain usable.
    legacy = Ref{Any}((multipliers=Matrix{Float64}(I,D,D),penalty=3.0))
    result = E.ladmmSolve(nothing,dims,H,sub,[0.7,0.3],0.1,chi,expired;warm_state=legacy)
    @test legacy[].multipliers == Matrix{ComplexF64}(I,D,D)
    @test legacy[].penalty == 3.0
    @test result[3] ≈ 0.9
    bad = Ref{Any}(merge(state[],(dims=[4],)))
    @test_throws DimensionMismatch E.ladmmSolve(nothing,dims,H,sub,[0.7,0.3],0.1,
        chi,expired;warm_state=bad)
    bad[] = merge(state[],(point=Float64[1],))
    @test_throws DimensionMismatch E.ladmmSolve(nothing,dims,H,sub,[0.7,0.3],0.1,
        chi,expired;warm_state=bad)
    # Validate the multiplier first, without reading missing point metadata.
    bad[] = (multipliers=zeros(ComplexF64,2,2),penalty=1.0,point=Float64[1])
    @test_throws DimensionMismatch E.ladmmSolve(nothing,dims,H,sub,[0.7,0.3],0.1,
        chi,expired;warm_state=bad)

    # Returned residuals use the normalised returned mixture even when the
    # time budget is exhausted before the first manifold retraction.
    tiny = [1-5e-10,5e-10]
    target = foldl(kron,[zero*zero',zero*zero'])
    result = E.ladmmSolve(nothing,dims,target,sub,tiny,1.0,chi,expired)
    @test length(result[5]) == 2
    @test result[5][2] > 0
    @test result[4] ≈ norm(mixture(result)-target) atol=1e-15
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

@testset "Fresh real CP seeds preserve conjugate groups" begin
    dims = [2,2]
    zero = ComplexF64[1,0]; one = ComplexF64[0,1]
    xp = (zero+one)/sqrt(2); xm = (zero-one)/sqrt(2)
    yp = (zero+im*one)/sqrt(2); ym = conj.(yp)
    sub = [[zero,zero],[one,one],[xp,xp],[xm,xm],[yp,ym],[ym,yp]]
    pure = [foldl(kron,[v*v' for v in s]) for s in sub]
    a = 0.95; b = sqrt(1-a^2)
    psi = ComplexF64[a,0,0,b]; H = psi*psi'
    upper = 4a*b/(1+4a*b)
    weights = [(1-upper)*a^2,(1-upper)*b^2,fill(upper/4,4)...]
    mixed = Matrix{ComplexF64}(I,4,4)/4
    @test sum(weights[j]*pure[j] for j in eachindex(weights)) ≈
        (1-upper)*H+upper*mixed atol=1e-14
    chi = Dict(:RE=>zeros(4,4),:IM=>zeros(4,4))
    param = Param(cp_real_master=true,ir_refit_scalar=true)
    w,s,_,_ = E.irWarmStart(pure,sub,weights,4,H,upper,chi,param)
    Y = sum(w[j]*foldl(kron,[v*v' for v in s[j]]) for j in eachindex(w))
    @test norm(imag(Y)) > 0.1
    wf,sf,zf,_ = E.irWarmStart(pure,sub,weights,4,H,upper,chi,param;fresh_cp=true)
    Yf = sum(wf[j]*foldl(kron,[v*v' for v in sf[j]]) for j in eachindex(wf))
    @test norm(imag(Yf)) <= 1e-14
    @test length(wf) == length(sf) <= 4
    @test sum(wf) ≈ 1
    @test all(norm(v) ≈ 1 for s in sf for v in s)
    B = H-mixed
    @test zf ≈ clamp(real(dot(B,Yf-mixed))/real(dot(B,B)),0.0,1.0)
    wg,sg,Pg = E.selectConjugateFactors(weights,4,pure,sub)
    @test wg == wf && sg == sf
    @test all(Pg[j] ≈ foldl(kron,[v*v' for v in sg[j]]) for j in eachindex(wg))

    # A pair of combined mass 0.6 dominates an isolated mass-0.4 column
    # under rank two; an all-pair rank-one input remains nonempty.
    wp,sp,Pp = E.selectConjugateFactors([0.4,0.3,0.3],2,
        pure[[1,5,6]],sub[[1,5,6]])
    @test wp ≈ [0.5,0.5]
    @test Pp == pure[[5,6]] && sp == sub[[5,6]]
    w1,s1,P1 = E.selectConjugateFactors([0.5,0.5],1,pure[[5,6]],sub[[5,6]])
    @test length(w1) == length(s1) == length(P1) == 1
    @test w1 == [1.0]
    # Unequal weights and unrelated adjacent complex states are singletons.
    for (packet,packet_sub,packet_weights) in (
        (pure[[1,5,6]],sub[[1,5,6]],[0.4,0.35,0.25]),
        (pure[[1,5,3]],sub[[1,5,3]],[0.4,0.3,0.3]))
        @test E.selectConjugateFactors(packet_weights,2,packet,packet_sub) ==
            E.selectTopFactors(packet_weights,2,packet_sub,packet)
    end
    _,_,z,_ = E.irWarmStart(pure,sub,weights,4,H,upper,chi,param;refit_scalar=false)
    @test z == 1-upper
    @test param.ir_refit_scalar
end

@testset "Fresh CP sign convention gives the correct scalar minimizer" begin
    dims = [2,2]
    zero = ComplexF64[1,0]; one = ComplexF64[0,1]
    sub = [[zero,zero],[one,one]]
    pure = [foldl(kron,[v*v' for v in s]) for s in sub]
    weights = [0.7,0.3]
    psi = ComplexF64[1,0,0,1]/sqrt(2); H = psi*psi'
    sigma = Matrix{ComplexF64}(I,4,4)/4
    B = H-sigma
    witness = B/real(dot(B,B))
    @test real(dot(witness,B)) ≈ 1
    chi = -witness
    Y = sum(weights[j]*pure[j] for j in eachindex(weights))
    penalty = 7.0
    quadratic = penalty*real(dot(B,B))
    linear = -1-real(dot(chi,B))-2penalty*real(dot(B,Y-sigma))
    scalar,_ = E.minimizeQuadraticOnUnitInterval(quadratic,linear,0.0)
    least_squares = clamp(real(dot(B,Y-sigma))/real(dot(B,B)),0.0,1.0)
    @test scalar ≈ least_squares atol=1e-14
    _,_,refitted,_ = E.irWarmStart(pure,sub,weights,2,H,0.25,
        Dict(:RE=>real(chi),:IM=>imag(chi)),Param(ir_refit_scalar=true))
    @test refitted ≈ scalar atol=1e-14
end

@testset "CP persistence and lazy pools keep aligned active columns" begin
    dims = [2,2]; D = prod(dims)
    zero = ComplexF64[1,0]; one = ComplexF64[0,1]
    complex_factor = (zero+im*one)/sqrt(2)
    product = [complex_factor,zero]; real_product = [one,one]
    P = foldl(kron,[v*v' for v in product])
    Q = foldl(kron,[v*v' for v in real_product])
    detector = E.ThresholdEntanglementDetector(zeros(D,D),zeros(D,D),dims,[],[])
    detector.model = E.Model()
    detector.M = Dict(:RE=>zeros(E.AffExpr,D,D),:IM=>zeros(E.AffExpr,D,D))
    detector.b = E.AffExpr()
    local_state = Dict(:RE=>[real(v*v') for v in product],
        :IM=>[imag(v*v') for v in product])
    E.addRank1PrincipleState(detector,local_state,
        Dict(:RE=>zeros(D,D),:IM=>zeros(D,D)),-1.0,true)
    @test detector.persistentInds == [1]
    E.clearStates(detector,false)
    @test length(detector.purestates) == length(detector.substates) ==
        length(detector.ispersistent) == 1
    E.clearStates(detector,true)
    @test isempty(detector.persistentInds) && isempty(detector.ispersistent)
    E.addBatchStates(detector,[Q],[real_product])
    E.clearStates(detector,false)
    @test isempty(detector.purestates) && isempty(detector.substates) &&
        isempty(detector.ispersistent) && isempty(detector.persistentInds)

    for realmaster in (false,true)
        detector = E.ThresholdEntanglementDetector(zeros(D,D),zeros(D,D),dims,[P],[product])
        detector.model = E.Model()
        detector.M = Dict(:RE=>zeros(E.AffExpr,D,D),:IM=>zeros(E.AffExpr,D,D))
        detector.b = E.AffExpr()
        detector.realmaster = realmaster
        E.addCons(detector,P)
        detector.poolpurestates = [P,Q,Q,conj.(P)]
        detector.poolsubstates = [product,real_product,real_product,[conj.(v) for v in product]]
        detector.poolstats = fill(1,4)
        detector.round = 2
        E.poolAdd(detector,Param())
        expected = realmaster ? 2 : 3
        @test length(detector.purestates) == length(detector.substates) ==
            length(detector.ispersistent) == length(detector.cuts) == expected
        @test detector.purestates[1] == P && detector.purestates[2] == Q
        @test realmaster || detector.purestates[3] == conj.(P)
        @test all(detector.purestates[j] ≈
            foldl(kron,[v*v' for v in detector.substates[j]]) for j in 1:expected)
    end
end
