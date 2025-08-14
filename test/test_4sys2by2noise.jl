
import Ket
using LinearAlgebra
using ExactEntanglement
# Problem setting
@testset "StateSeparator test noisy 2by2by2by2 matrix" begin
    p = 1 - 1.0/9 # 1 - 1/9
    T = Float64
    dims = [2, 2, 2, 2]
    target_rho = Hermitian((1-p) * Ket.state_ghz(2, 4) + p*Complex.(Matrix{T}(I, prod(dims), prod(dims))/prod(dims)))
    dims = [2, 2, 2, 2]
    target_rho = Matrix(target_rho)
    print(typeof(target_rho))
    HR = real(target_rho)
    HI = imag(target_rho)

    param = Param("MSK")
    #problem = buildProblemRecursiveSingle(HR, HI, [2, 2, 2], 1.0)
    problem = detectEntanglementProjective(HR, HI, dims, param)
    #problem = detectEntanglementThreshold(HR, HI, dims, param)
    #problem = detectEntanglementHybrid(HR, HI, dims, param)

    # Some oracle give us a guess
end