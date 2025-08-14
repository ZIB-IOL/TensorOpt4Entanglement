
using Ket
using LinearAlgebra
using Random
using ExactEntanglement
# Problem setting
@testset "StateSeparator test bipartite matrix" begin
    T = Float64
    N = 3
    dims = Tuple(fill(2, N))
    # GHZ state 2x...x2
    # the GME threshold is known: doi:10.1088/1367-2630/12/5/053002.
    v = 0.42857 - 0.01
    ρ = state_ghz(Complex{T}, 2, N; v)

    HR = real(ρ)
    HI = imag(ρ)
    param = Param("MSK")
    #problem = buildProblemRecursiveSingle(HR, HI, [2, 2, 2], 1.0)
    #problem = detectEntanglementProjective(HR, HI, dims, param)
    #problem = detectEntanglementThreshold(HR, HI, dims, param)
    problem = detectEntanglementHybrid(HR, HI, dims, param)

    # Some oracle give us a guess
end