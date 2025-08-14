using Ket
using LinearAlgebra
using Random
using ExactEntanglement
#using EntanglementDetection
# Problem setting
@testset "StateSeparator test bipartite matrix" begin
    T = Float64
    CT = Complex{T}

    # The threshold == 1 - 1 / (1 + 2^(N-1)) for ghz states
    N = 4
    dims = fill(2, N)
    Random.seed!(0)
    ρ = state_dicke(CT, N ÷ 2, N)

    ρ = Matrix(ρ)
    HR = real(ρ)
    HI = imag(ρ)
    param = Param("MSK")
    #problem = buildProblemRecursiveSingle(HR, HI, [2, 2, 2], 1.0)
    #problem = detectEntanglementProjective(HR, HI, dims, param)
    problem = detectEntanglementThresholdLiftDiscrete(HR, HI, dims, param)
    #problem = detectEntanglementHybrid(HR, HI, dims, param)

    # Some oracle give us a guess
end
