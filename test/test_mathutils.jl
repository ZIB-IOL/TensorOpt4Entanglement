using Test, LinearAlgebra, Random
const E = ExactEntanglement

@testset "MathUtils" begin
    @test E.cumulativeAdd!([2, 3, 4]) == [2, 5, 9]

    @test E.unravelIndex(0, [2, 3]) == [1, 1]
    @test E.unravelIndex(5, [2, 3]) == [2, 3]
    # unravelIndex and cartIndex are inverse views of the same layout
    for d in ([2, 3], [2, 2, 2], [3, 2])
        for k in 0:prod(d)-1
            @test E.unravelIndex(k, d) == E.cartIndex(d, k + 1)
        end
    end

    @testset "minimiser of a quadratic on [0,1]" begin
        # interior minimum
        z, f = E.minimizeQuadraticOnUnitInterval(1.0, -1.0, 0.0)
        @test z ≈ 0.5 && f ≈ -0.25
        # linear, decreasing -> right endpoint
        @test E.minimizeQuadraticOnUnitInterval(0.0, -2.0, 1.0)[1] == 1.0
        # linear, increasing -> left endpoint
        @test E.minimizeQuadraticOnUnitInterval(0.0, 2.0, 1.0)[1] == 0.0
        # the returned value really is the minimum over a fine grid
        for (a, b, c) in ((1.0, -0.5, 0.2), (3.0, 1.0, -1.0), (-1.0, 0.3, 0.0))
            z, f = E.minimizeQuadraticOnUnitInterval(a, b, c)
            grid = minimum(a * t^2 + b * t + c for t in 0:1e-4:1)
            @test f <= grid + 1e-6
        end
    end

    @test E.realInner([1.0 + 2im], [3.0 - 1im]) ≈ 3 * 1 + 2 * (-1)

    @testset "projectDensityMat" begin
        h = [2.0 0.0; 0.0 1.0]
        P = E.projectDensityMat(h)
        @test tr(P) ≈ 1
        @test P ≈ [1.0 0.0; 0.0 0.0]          # projects onto the top eigenvector
        @test P ≈ P'                           # Hermitian
    end

    @testset "buildIndexMap" begin
        dims = [2, 2]
        m = E.buildIndexMap(dims)
        @test size(m) == (4, 4)
        # every entry references each mode once from the row and once from the column
        @test all(length(m[i, j]) == 2 * length(dims) for i in 1:4, j in 1:4)
        @test all(sum(t[3] for t in m[i, j]) == 0 for i in 1:4, j in 1:4)
    end

    @testset "partialInnerProductMap" begin
        # result[i,j] = <A (x) E_ij (x) C, D> with the Hermitian inner product
        # (conjugate-linear in the first argument). Checked against the naive
        # definition, which is what the original scratch script verified.
        rng = MersenneTwister(2024)
        for (m, n, p) in ((2, 2, 2), (2, 3, 2), (3, 2, 2))
            A = randn(rng, ComplexF64, m, m); A = (A + A') / 2
            C = randn(rng, ComplexF64, p, p); C = (C + C') / 2
            D = randn(rng, ComplexF64, m * n * p, m * n * p)
            fast = E.partialInnerProductMap(A, C, D)
            @test size(fast) == (n, n)
            naive = zeros(ComplexF64, n, n)
            for i in 1:n, j in 1:n
                Eij = zeros(n, n); Eij[i, j] = 1.0
                naive[i, j] = dot(kron(A, Eij, C), D)
            end
            @test fast ≈ naive
        end
    end
end
