using Test, LinearAlgebra
const E = ExactEntanglement

@testset "sBB structures" begin
    for dims in ([2, 2, 2], [2, 3, 2, 2])
        d = prod(dims)
        p = E.Problem(Matrix(Diagonal(ones(d))) / d, zeros(d, d), dims)

        @testset "bipartition tree $dims" begin
            @test p.dimH == d
            @test p.nsubs == length(dims)
            # Z indices are assigned in post-order, so the root is the last one
            @test p.Zrootid == length(p.Zdims)
            @test p.Zdims[p.Zrootid] == d
            @test length(p.Zdims) == 2 * length(dims) - 1
            @test length(p.Zleafids) == length(dims)   # one leaf per subsystem
            @test sort([p.Zdims[i] for i in p.Zleafids]) == sort(dims)
            # every internal node's dimension is the product of its children's
            for sys in p.BST
                if sys.sysid == -1 && sys.left != -1
                    @test p.Zdims[sys.Zind] ==
                          p.Zdims[p.BST[sys.left].Zind] * p.Zdims[p.BST[sys.right].Zind]
                end
            end
        end

        @testset "root node bounds $dims" begin
            root = E.createRootNode(p.dims, p.BST, p.Zdims, 1e-6, p.globalZBs, p.fixvars)
            @test length(root.ZBs) == length(p.Zdims)
            # the imaginary part is anti-symmetric, so its diagonal is pinned to 0
            @test length(root.fixvars) == sum(p.Zdims)
            @test all(k[2] === :IM && k[3] == k[4] for k in keys(root.fixvars))
            @test all(v == 0.0 for v in values(root.fixvars))
            # bounds bracket every admissible density-matrix entry
            for (i, ZB) in enumerate(root.ZBs)
                @test all(ZB[(:RE, :L)] .<= ZB[(:RE, :U)])
                @test all(ZB[(:IM, :L)] .<= ZB[(:IM, :U)])
                @test all(abs(ZB[:RE,:L][j,k]) <= 0.5+1e-6 &&
                    abs(ZB[:RE,:U][j,k]) <= 0.5+1e-6 for j in 1:p.Zdims[i] for k in 1:p.Zdims[i] if j != k)
            end
        end
    end

    @testset "McCormick bound gathering honours fixed vars" begin
        ZBs = [Dict((:RE, :L) => fill(-1.0, 2, 2), (:RE, :U) => fill(1.0, 2, 2),
                    (:IM, :L) => fill(-1.0, 2, 2), (:IM, :U) => fill(1.0, 2, 2)) for _ in 1:2]
        b = E.subFactorBounds(Dict((1, :RE, 1, 1) => 0.25), ZBs, (1, 2), [1, 1], [1, 1])
        @test b[:RE, :L][1] == 0.25 && b[:RE, :U][1] == 0.25   # fixed -> collapsed
        @test b[:RE, :L][2] == -1.0 && b[:RE, :U][2] == 1.0    # free  -> node bounds
    end
end

@testset "sBB contractions and complete coordinate sweeps" begin
    rng = E.MersenneTwister(9876)
    for (m,n,p) in ((1,2,3), (2,3,2), (3,2,1))
        A = randn(rng,ComplexF64,m,m)
        C = randn(rng,ComplexF64,p,p)
        D = randn(rng,ComplexF64,m*n*p,m*n*p)
        G = E.partialInnerProductMap(A,C,D)
        for i in 1:n, j in 1:n
            basis = zeros(n,n); basis[i,j] = 1
            @test G[i,j] ≈ dot(kron(A,basis,C),D)
        end
    end
    # The first updated factor is stationary, but the first subsystem isn't.
    H = kron([1.0 0; 0 0],Matrix{Float64}(I,4,4))
    ss = E.StateSeparator(E.Problem(H,zeros(8,8),[2,2,2]),Param(log_level=0))
    X = fill(0.5,2,2)
    start = Dict(:RE=>[copy(X) for _ in 1:3], :IM=>[zeros(2,2) for _ in 1:3])
    E.AlternativeDescentEigen(ss,start)
    @test ss.primalbd ≈ 1.0
    @test E.constructFullSol(ss.primalsol) ≈ ss.primalHbar
end

@testset "sBB splits preserve Hermitian boxes and prune descendants" begin
    problem = E.Problem(Matrix{Float64}(I,8,8)/8,zeros(8,8),[2,2,2])
    for part in (:RE,:IM)
        ss = E.StateSeparator(problem,Param(log_level=0))
        root = E.createRootNode(problem.dims,problem.BST,problem.Zdims,1e-6,[],problem.fixvars)
        E.stateseparatorAddNode!(ss,root)
        lower = root.ZBs[1][part,:L][1,2]
        upper = root.ZBs[1][part,:U][1,2]
        root.localdualbd = 0.3
        E.stateseparatorCreateBranchNodes!(ss,root,part,1,1,2,lower)
        down,up = ss.nodes[2:3]
        split = down.ZBs[1][part,:U][1,2]
        @test lower < split < upper
        @test split == up.ZBs[1][part,:L][1,2]
        @test down.ZBs[1][part,:L][1,2] == lower
        @test up.ZBs[1][part,:U][1,2] == upper
        for child in (down,up)
            L = child.ZBs[1][part,:L]; U = child.ZBs[1][part,:U]
            @test part === :RE ? L ≈ L' && U ≈ U' : L ≈ -U'
        end
        # A dominated parent certifies that all of its descendants can be pruned.
        ss.primalbd = 0.4
        E.stateseparatorUpdateTree!(ss)
        @test all(n -> n.pruned,ss.nodes)
        @test isempty(ss.leaves)
        @test ss.dualbd >= ss.primalbd
    end
end

@testset "Global OBBT excludes witness-dependent restrictions" begin
    problem = E.Problem(Matrix{Float64}(I,8,8)/8,zeros(8,8),[2,2,2])
    plain = E.buildRelaxationBound(problem,0.0,1,:RE,1,1,:L,true)
    problem.proximal = Dict(:RE=>zeros(8,8),:IM=>zeros(8,8))
    global_model = E.buildRelaxationBound(problem,0.0,1,:RE,1,1,:L,true)
    local_model = E.buildRelaxationBound(problem,0.0,1,:RE,1,1,:L,false)
    @test E.modelStats(plain.model)[2] == E.modelStats(global_model.model)[2]
    @test E.modelStats(local_model.model)[2] > E.modelStats(global_model.model)[2]
    root = E.createRootNode(problem.dims,problem.BST,problem.Zdims,1e-6,[],problem.fixvars)
    E.tightenEntryBounds!(root,1,:IM,1,2,-0.2,0.3)
    E.tightenEntryBounds!(root,1,:IM,1,2,-0.8,0.7) # weaker dual bounds cannot widen it
    @test root.ZBs[1][:IM,:L][1,2] == -0.2
    @test root.ZBs[1][:IM,:U][1,2] == 0.3
    @test root.ZBs[1][:IM,:L][2,1] == -0.3
    @test root.ZBs[1][:IM,:U][2,1] == 0.2
end

@testset "Certifying relaxations retain products excluded by proximal cuts" begin
    # A normalised GHZ witness perturbed along X ⊗ Z ⊗ |0><0|. Its maximum
    # with the last two factors fixed at |0> is positive, but the old proximal
    # half-space excludes that maximising product.
    v = zeros(ComplexF64,8); v[1] = v[8] = inv(sqrt(2))
    H = v*v'; mixed = Matrix{ComplexF64}(I,8,8)/8
    M = zeros(ComplexF64,8,8); M[1,8] = M[8,1] = 0.8
    for i in 2:7; M[i,i] = -0.8/3; end
    X = ComplexF64[0 1;1 0]; Z = ComplexF64[1 0;0 -1]; P = ComplexF64[1 0;0 0]
    M += 0.02 * kron(X,Z,P)
    _,U = eigen(Hermitian(ComplexF64[0 0.02;0.02 -0.8/3]))
    winner = kron(U[:,end]*U[:,end]',P,P)
    proximal = 0.2H + 0.8mixed
    @test real(dot(M,H-mixed)) ≈ 1
    @test real(dot(M,winner)) > 0
    @test real(dot(winner-proximal,M-proximal)) < -0.03
    problem = E.Problem(real(M),imag(M),[2,2,2])
    plain = E.buildRelaxationTrivial(problem,0.0)
    problem.proximal = Dict(:RE=>real(proximal),:IM=>imag(proximal))
    with_proximal = E.buildRelaxationTrivial(problem,0.0)
    @test E.modelStats(plain.model)[2] == E.modelStats(with_proximal.model)[2]
end

@testset "Branching scores use the scalar McCormick envelopes" begin
    rng = E.MersenneTwister(456)
    for trial in 1:40
        vars = Dict(:RE=>randn(rng,2), :IM=>randn(rng,2))
        bounds = Dict((part,dir)=>randn(rng,2) for part in (:RE,:IM) for dir in (:L,:U))
        for part in (:RE,:IM), k in 1:2
            l,u = minmax(bounds[part,:L][k],bounds[part,:U][k])
            bounds[part,:L][k] = l
            bounds[part,:U][k] = trial % 5 == 0 ? l : u
        end
        z = Dict(:RE=>randn(rng), :IM=>randn(rng))
        costs = Dict(:RE=>randn(rng), :IM=>randn(rng))
        lower,upper = Dict(),Dict()
        for a in (:RE,:IM), b in (:RE,:IM)
            lower[a,b] = maximum(E.affine(vars[a][1],vars[b][2],
                bounds[a,:L][1],bounds[b,:L][2],bounds[a,:U][1],bounds[b,:U][2]))
            upper[a,b] = minimum(E.affine(vars[a][1],vars[b][2],
                bounds[a,:U][1],bounds[b,:L][2],bounds[a,:L][1],bounds[b,:U][2]))
        end
        L = Dict(:RE=>lower[:RE,:RE]-upper[:IM,:IM], :IM=>lower[:IM,:RE]+lower[:RE,:IM])
        U = Dict(:RE=>upper[:RE,:RE]-lower[:IM,:IM], :IM=>upper[:IM,:RE]+upper[:RE,:IM])
        @test E.branchViolation(vars,bounds,z) ≈ sum(max(L[a]-z[a],0)+max(z[a]-U[a],0) for a in (:RE,:IM))
        @test E.branchViolation(vars,bounds,z,costs) ≈ sum((max(L[a]-z[a],0)-max(z[a]-U[a],0))*costs[a] for a in (:RE,:IM))
    end
end

@testset "Complex McCormick includes crossed affine bounds" begin
    problem = E.Problem(Matrix{Float64}(I,4,4)/4,zeros(4,4),[2,2])
    ss = E.StateSeparator(problem,Param(log_level=0))
    root = E.createRootNode(problem.dims,problem.BST,problem.Zdims,0.0,[],problem.fixvars)
    opt = E.buildBaseRelaxation(problem)
    E.addComplexMcCormickConstraints(ss,opt,root)
    # For off-diagonal factors with bounds [-1/2,1/2], combine the second
    # lower bound of Re(x)Re(y) with the first upper bound of Im(x)Im(y).
    expected = opt.Y[:RE][1,4] - 0.5opt.Xs[:RE][1][1,2] - 0.5opt.Xs[:RE][2][1,2] +
        0.5opt.Xs[:IM][1][1,2] - 0.5opt.Xs[:IM][2][1,2]
    constraints = E.JuMP.all_constraints(opt.model,E.JuMP.AffExpr,E.MOI.GreaterThan{Float64})
    @test any(constraints) do c
        obj = E.JuMP.constraint_object(c)
        obj.set.lower == -0.5 && E.JuMP.isequal_canonical(obj.func,expected)
    end
    # x=0.3-0.3i, y=0.3+0.3i gives lower bound 0.1. The old zipped
    # combination gave -0.5 and thus admitted, for example, Re(z)=-0.1.
    @test real((0.3-0.3im)*(0.3+0.3im)) >= 0.1
end

@testset "Node-guided heuristics keep a feasible improving incumbent" begin
    rng = E.MersenneTwister(876)
    A = randn(rng,ComplexF64,8,8); H = Matrix(Hermitian(A+A'))
    problem = E.Problem(real(H),imag(H),[2,2,2])
    relaxed = Dict(:RE=>[Matrix{Float64}(I,2,2)/2 for _ in 1:3],:IM=>[zeros(2,2) for _ in 1:3])
    single = E.StateSeparator(problem,Param(log_level=0,heur_sbb_node_restarts=1))
    multi = E.StateSeparator(problem,Param(log_level=0,heur_sbb_node_restarts=4))
    E.RunHeuristics(single,(;Xvals=relaxed))
    E.RunHeuristics(multi,(;Xvals=relaxed))
    @test multi.primalbd >= single.primalbd - 1e-12
    @test multi.primalHbar ≈ E.constructFullSol(multi.primalsol)
    @test multi.primalbd ≈ real(dot(H,multi.primalHbar))
    for (R,I) in zip(multi.primalsol[:RE],multi.primalsol[:IM])
        @test tr(R) ≈ 1
        @test minimum(eigvals(Hermitian(R+im*I))) >= -1e-12
    end
    @test relaxed[:RE][1] == Matrix{Float64}(LinearAlgebra.I,2,2)/2
end

@testset "sBB bounds survive interruption" begin
    dims=[2,2,2]; D=prod(dims)
    p=Param(log_level=0)
    problem=E.Problem(Matrix{Float64}(I,D,D)/D,zeros(D,D),dims)
    problem.cutoffbound=0.2
    ss=E.StateSeparator(problem,p)
    root=E.createRootNode(dims,problem.BST,problem.Zdims,p.feas_tol,[],problem.fixvars)
    @test root.localdualbd==Inf
    E.stateseparatorAddNode!(ss,root)
    root.localdualbd=1.2
    E.stateseparatorCreateBranchNodes!(ss,root,:RE,1,1,1,0.5)
    @test all(ss.nodes[i].localdualbd==1.2 for i in (2,3))
    E.stateseparatorUpdateTree!(ss)
    @test ss.dualbd==1.2
    ss.nodes[2].pruned=ss.nodes[3].pruned=true
    E.stateseparatorUpdateTree!(ss)
    @test ss.dualbd>=problem.cutoffbound

    expired=Param(time_limit=1.0,start_time=time()-2,log_level=0)
    @test E.threshold!(problem,expired)==0.0
    primal,dual,Hbar,sol=E.separate!(problem,expired)
    @test primal==-Inf && dual==Inf && isnothing(Hbar) && isnothing(sol)
end

@testset "sBB heuristic factors match the winning product" begin
    dims=[2,2,2]; D=prod(dims)
    H=Matrix{Float64}(I,D,D)/D
    ss=E.StateSeparator(E.Problem(H,zeros(D,D),dims),Param(time_limit=3600.0,log_level=0))
    X=fill(0.5,2,2)
    seed=Dict(:RE=>[copy(X) for _ in dims],:IM=>[zeros(2,2) for _ in dims])
    E.AlternativeDescentEigen(ss,seed)
    @test E.constructFullSol(ss.primalsol)≈ss.primalHbar
    @test ss.primalbd≈dot(real(ss.primalHbar),H)
end
