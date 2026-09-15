using Test, LinearAlgebra
const E = ExactEntanglement

@testset "every benchmark instance loads" begin
    dir = joinpath(@__DIR__, "..", "benchmark")
    files = sort(filter(f -> endswith(f, ".jl"), readdir(dir)))
    @test !isempty(files)

    for f in files
        @testset "$f" begin
            # loadBenchmark evaluates the file in a sandbox module; the files
            # rely on LinearAlgebra being ambiently available (e.g. `norm`),
            # which a bare Module() does not provide.
            d = E.loadBenchmark(f)
            @test length(d.dims) == d.N
            rho = Matrix(d.ρ)
            @test size(rho, 1) == prod(d.dims)
            @test size(rho, 1) == size(rho, 2)
            @test isapprox(tr(rho), 1; atol = 1e-8)        # a density matrix
            @test isapprox(rho, rho'; atol = 1e-8)         # Hermitian
            @test minimum(real(eigvals(Hermitian(rho)))) > -1e-8   # PSD
        end
    end
end
