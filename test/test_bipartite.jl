
using Ket
using LinearAlgebra
using Random
using ExactEntanglement
# Problem setting
@testset "StateSeparator test bipartite matrix" begin
    T = Float64
    N = 2
    dims = [10, 2]
    Random.seed!(0)
    ρ = random_state(Complex{T}, prod(dims), 1)
    v = 0.2
    white_noise!(ρ, v)

    ρ = Matrix(ρ)
    HR = real(ρ)
    HI = imag(ρ)
    param = Param("MSK")


    #problem = buildProblemRecursiveSingle(HR, HI, [2, 2, 2], 1.0)
    problem = detectEntanglementProjective(HR, HI, dims, param)
    #problem = detectEntanglementThreshold(HR, HI, dims, param)
    #problem = detectEntanglementHybrid(HR, HI, dims, param)

    # Some oracle give us a guess
end