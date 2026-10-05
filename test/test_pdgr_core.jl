using Test, LinearAlgebra, Random
using ExactEntanglement: PDGR

const SMALL = PDGR.Options(max_steps=2, fw_max_iteration=200,
    witness_max_length=1000, lmo_nb=2, lmo_max_iter=30, verbose=0)

@testset "PDGR numerical core" begin
    psi = ComplexF64[1, 0, 0, 1]
    rho = psi * psi' / real(dot(psi, psi))

    @testset "Bell threshold and returned certificates" begin
        result = PDGR.solve(rho, [2, 2]; options=SMALL)
        @test 0.4 < result.ent_bound <= 2/3 + 1e-9
        @test 2/3 - 1e-9 <= result.sep_bound < 0.9
        @test result.gap ≈ result.sep_bound - result.ent_bound
        @test length(result.history) == 2
        @test result.witness !== nothing
        @test result.witness.valid
        W = result.witness.W
        noisy = (1-result.ent_bound) * rho + result.ent_bound * I / 4
        @test real(dot(W, noisy)) <= 1e-10
        rng = MersenneTwister(42)
        for _ in 1:20
            product = kron(normalize(randn(rng, ComplexF64, 2)),
                           normalize(randn(rng, ComplexF64, 2)))
            @test real(dot(product, W * product)) >= -1e-10
        end
        cert = result.separability_certificate
        @test cert.source == "geometric_reconstruction"
        @test eigmin(Hermitian(cert.σ)) >= -1e-10
        @test real(tr(cert.σ)) ≈ 1
        @test result.sep_bound ≈ (cert.probe + cert.distance/cert.radius) /
                                (1 + cert.distance/cert.radius)
        # Matrix and normalized-ket entry points must agree with the same seed.
        from_ket = PDGR.solve(3psi, (2, 2); options=SMALL)
        @test from_ket.ent_bound ≈ result.ent_bound atol=1e-10
        @test from_ket.sep_bound ≈ result.sep_bound atol=1e-10
        # A later, weaker witness must not overwrite the certificate for the
        # stored best lower bound (including near-degenerate rounded inputs).
        rounded = rho * (1 - eps())
        again = PDGR.solve(rounded, (2, 2); options=SMALL)
        boundary = (1-again.ent_bound)*rounded + again.ent_bound*I/4
        @test real(dot(again.witness.W, boundary)) <= 1e-10
    end

    @testset "separable input and one-sided modes" begin
        product = ComplexF64[1, 0, 0, 0]
        result = PDGR.solve(product, (2, 2); options=SMALL)
        @test result.ent_bound == 0
        @test result.sep_bound < 1e-5
        for mode in (:sep, :ent)
            one_sided = PDGR.solve(rho, (2, 2); options=SMALL, mode)
            @test one_sided.gap === nothing
            @test (mode == :sep ? one_sided.ent_bound : one_sided.sep_bound) === nothing
        end
        mixed = PDGR.solve(Matrix{ComplexF64}(I, 4, 4)/4, (2, 2); options=SMALL)
        @test mixed.ent_bound == 0
        @test mixed.sep_bound < 0.05
    end

    @testset "timeouts preserve safe bounds" begin
        result = PDGR.solve(rho, (2, 2); options=SMALL, time_limit=0)
        @test result.status == "time_limit"
        @test result.ent_bound == 0
        @test result.sep_bound == 1
        @test result.witness === nothing
        @test isempty(result.history)
        core = PDGR.EntanglementDetection
        lmo = core.EnumeratingSeparableLMO(Float64, (2, 2); max_length=1000)
        direction = core.correlation_tensor(rho, (2, 2))
        task_local_storage(:PDGR_deadline, -Inf) do
            @test_throws core.TimeLimitReached core.FrankWolfe.compute_extreme_point(lmo, direction)
        end
        @test !haskey(task_local_storage(), :PDGR_deadline)
        @test PDGR.solve(rho, (2, 2); options=SMALL).ent_bound > 0
    end

    @testset "validation" begin
        @test_throws ArgumentError PDGR.solve(rho, (2, 3); options=SMALL)
        @test_throws ArgumentError PDGR.solve(2rho, (2, 2); options=SMALL)
        @test_throws ArgumentError PDGR.solve(Diagonal([1.1, -0.1, 0, 0]), (2, 2); options=SMALL)
        @test_throws ArgumentError PDGR.solve(zeros(4), (2, 2); options=SMALL)
        @test_throws ArgumentError PDGR.solve(rho, (2, 2); options=SMALL, mode=:invalid)
        @test_throws ArgumentError PDGR.solve(rho, (2, 2); options=SMALL, structure=3)
        @test_throws ArgumentError PDGR.solve(rho, (2, 2); options=SMALL, max_steps=0)
        @test_throws ArgumentError PDGR.solve(rho, (2, 2); options=SMALL, time_limit=-1)
        @test_throws ArgumentError PDGR.solve(rho, (2, 2); options=SMALL, fw_epsilon=NaN)
        @test_throws ArgumentError PDGR.solve(rho, (2, 2); options=SMALL, witness_max_length=1)
    end
end
