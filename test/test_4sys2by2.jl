
import Ket
using Random

# Problem setting
@testset "StateSeparator test 2by2by2by2 separable state" begin
    dims = [2, 2, 2, 2]
    seed = MersenneTwister(100)
    Vvals = [ rand(seed, dim, dim) + im * rand(seed, dim, dim)  for dim in dims]
    PSDs = [ V' * V  for V in Vvals]
    Xvals = [ PSD / tr(PSD) for PSD in PSDs]
    target_rho = foldl(kron, Xvals)
    target_rho = Matrix(target_rho)
    print(typeof(target_rho))
    HR = real(target_rho)
    HI = imag(target_rho)

    param = Param("MSK")
    #problem = buildProblemRecursiveSingle(HR, HI, [2, 2, 2], 1.0)
    problem = Problem(HR, HI, dims)
    solve(problem, param)

    # Some oracle give us a guess
end