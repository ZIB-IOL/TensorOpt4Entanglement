

# Test the greet() function
@testset "StateSeparator test 2by2by2 matrix" begin
    Bell = normalize([1, 0, 0, 0, 0, 0, 0, 1].+ 0 * im)
    p = 2/3
    #H = (1 - p ) * Bell*Bell'  + p*Matrix(I,4,4)/4
    H =  Bell*Bell'
    HR = real(H)
    HR = reshape(HR, 8, 8)
    HI = imag(H)
    HI = reshape(HI, 8, 8)
    param = Param("MSK")
    #problem = buildProblemRecursiveSingle(HR, HI, [2, 2, 2], 1.0)
    problem = Problem(HR, HI, [2, 2, 2])
    solve(problem, param)
end

