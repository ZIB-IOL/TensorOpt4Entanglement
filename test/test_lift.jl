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
        @test E.manifold_dimension(M) == 2 * sumdim * nr
        # weights fold into the vectors so that Psi(p) == sum_j w_j * p_j
        y = E.liftMap(M, p)
        ref = sum(w[j] * foldl(kron, [s * s' for s in subs[j]]) for j in 1:nr)
        @test y ≈ ref
        @test E.fastTrace(M, p) ≈ 1
        @test real(tr(y)) ≈ 1

        _, _, w2 = E.unpackFactors(p, r, sumdim, dims, cdims, nsubs)
        @test sort(w2) ≈ sort(w)
    end

    @testset "retraction keeps unit trace" begin
        g = randn(MersenneTwister(3), length(p))
        q = similar(p)
        E.retract_project!(M, q, p, 0.01 .* g)
        @test E.fastTrace(M, q) ≈ 1
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
    end
end
