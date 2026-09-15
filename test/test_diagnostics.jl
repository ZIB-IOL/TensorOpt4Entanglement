using Test, LinearAlgebra, JuMP
const E = ExactEntanglement

@testset "diagnostics and relaxation modes" begin
    @testset "functionNNZ" begin
        m = Model()
        @variable(m, x[1:3])
        @test E.functionNNZ(x[1]) == 1
        @test E.functionNNZ(2x[1] + 3x[2]) == 2
        @test E.functionNNZ(1.0) == 0
        @test E.functionNNZ([x[1] + x[2], x[3]]) == 3
        # complex affine expressions appear in the partial-trace constraints and
        # are a different concrete type from the real ones
        @test E.functionNNZ((1 + 2im) * x[1] + x[2]) == 2
    end

    @testset "DDPS is strictly weaker than DDPS+" begin
        for dims in ([2, 2, 2], [2, 2, 2, 2])
            d = prod(dims)
            HR = Matrix(Diagonal(ones(d))) / d
            HI = zeros(d, d)
            plain = E.relaxationStats(HR, HI, dims, Param(log_level = 0, relaxation = :ddps))
            plus  = E.relaxationStats(HR, HI, dims, Param(log_level = 0, relaxation = :ddpsplus))
            @test plain.nvars == plus.nvars            # same variables...
            @test plain.ncons < plus.ncons             # ...fewer constraints
            @test plain.nnz < plus.nnz
            @test plain.nvars > 0 && plain.nnz > 0
        end
    end

    @testset "DDPS+ is the default and is unchanged" begin
        @test Param().relaxation === :ddpsplus
        dims = [2, 2, 2]; d = prod(dims)
        s = E.relaxationStats(Matrix(Diagonal(ones(d))) / d, zeros(d, d), dims, Param(log_level = 0))
        # pinned: the root relaxation for 3 qubits
        @test s.nvars == 92
        @test s.ncons == 1040
    end

    @testset "peak RSS is reported" begin
        @test E.peakRSSMiB() > 0
    end

    @testset "every DDPS algorithm code has a DDPS+ counterpart" begin
        for code in ("RLT", "D", "LDL")
            @test haskey(E.ALGORITHMS, code)
            @test haskey(E.ALGORITHMS, code * "_DDPS")
        end
    end
end
