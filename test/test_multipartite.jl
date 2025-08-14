
using Ket
using LinearAlgebra
using Random
using ExactEntanglement
# Problem setting
@testset "StateSeparator test bipartite matrix" begin
    T = Float64
    N = 5
    dims = fill(2, N)
    # GHZ state 2x...x2
    # the threshold is known: https://arxiv.org/pdf/quant-ph/9911044
    v = 1 / (1 + 2^(N-1)) - 0.01
    print(1 / (1 + 2^(N-1)))
    ρ = state_ghz(Complex{T}, 2, N; v= 1)

    ρ = Matrix(ρ)
    HR = real(ρ)
    HI = imag(ρ)
    param = Param("MSK")
    #problem = buildProblemRecursiveSingle(HR, HI, [2, 2, 2], 1.0)
    #problem = detectEntanglementProjective(HR, HI, dims, param)
    problem = detectEntanglementThresholdEasy(HR, HI, dims, param)
    #problem = detectEntanglementHybrid(HR, HI, dims, param)

    # Some oracle give us a guess
end