
using Ket
using LinearAlgebra
using Random
using ExactEntanglement
# Problem setting
@testset "StateSeparator test bipartite matrix" begin
    T = Float64
    N = 2
    dims = [3, 3]
    # Horodecki state 3x3
    v = 0.98 # controls how the pure state is mixed with the center of the set of separable states
    p = [
        1,
        0.987,
        0.982,
        0.977,
        0.971,
        0.966,
        0.962,
        0.958,
        0.955,
        0.953,
        0.950,
        0.948,
        0.947,
        0.945,
        0.944,
        0.943,
        0.941,
        0.941,
        0.940,
        0.939,
        0.939,
        0.938,
        0.938,
        0.938,
        0.938,
        0.937,
        0.937,
        0.937,
        0.937,
        0.937,
        0.937,
        0.938,
        0.938,
        0.938,
        0.938,
        0.938,
        0.939,
        0.939,
        0.940,
        0.940,
        0.940,
        0.941,
        0.941,
        0.942,
        0.942,
        0.942,
        0.943,
        0.944,
        0.944,
        0.945,
        0.946,
        0.946,
        0.947,
        0.948,
        0.948,
        0.949,
        0.950,
        0.950,
        0.951,
        0.952,
        0.953,
        0.954,
        0.955,
        0.955,
        0.956,
        0.957,
        0.958,
        0.959,
        0.960,
        0.961,
        0.962,
        0.963,
        0.964,
        0.965,
        0.966,
        0.967,
        0.968,
        0.969,
        0.970,
        0.971,
        0.972,
        0.974,
        0.975,
        0.976,
        0.977,
        0.978,
        0.980,
        0.981,
        0.982,
        0.984,
        0.984,
        0.986,
        0.987,
        0.989,
        0.990,
        0.992,
        0.993,
        0.995,
        0.997,
        0.999,
        1
    ]
    print(1 - 0.9448701948222268)
    ρ = state_horodecki33(Complex{T}, 0.5049494949494949; v = 1)

    ρ = Matrix(ρ)
    HR = real(ρ)
    HI = imag(ρ)
    param = Param("MSK")
    #problem = buildProblemRecursiveSingle(HR, HI, [2, 2, 2], 1.0)
    #problem = detectEntanglementProjective(HR, HI, dims, param)
    problem = detectEntanglementThresholdHard(HR, HI, dims, param)
    #problem = detectEntanglementHybrid(HR, HI, dims, param)

    # Some oracle give us a guess
end