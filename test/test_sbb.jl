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
