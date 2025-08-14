using Test
using BoundEntangle
using LinearAlgebra

# Test the greet() function
@testset "BoundEntangle test 3by3 matrix" begin
   a = 0.2
   b = (1 + a) / 2
   c = sqrt(1 - a^2) / 2
   H = [
      a 0 0 0 a 0 0 0 a
      0 a 0 0 0 0 0 0 0
      0 0 a 0 0 0 0 0 0
      0 0 0 a 0 0 0 0 0
      a 0 0 0 a 0 0 0 a
      0 0 0 0 0 a 0 0 0
      0 0 0 0 0 0 b 0 c
      0 0 0 0 0 0 0 a 0
      a 0 0 0 a 0 c 0 b
   ] / (8 * a + 1)
   HR = real(H)
   HR = reshape(HR, 9, 9)
   HI = imag(H)
   HI = reshape(HI, 9, 9)
   option = Option("IPM", false)
   problem = buildProblemStandard(HR, HI, [3, 3], 1.0) 
   solveProblem(problem, option)
end