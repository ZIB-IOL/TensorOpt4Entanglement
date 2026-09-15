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

    @testset "every DDPS+ algorithm has a DDPS counterpart" begin
        for (plus, plain) in ("DDPS+" => "DDPS", "CP" => "CP-DDPS", "IR" => "IR-DDPS")
            @test haskey(E.ALGORITHMS, plus)
            @test haskey(E.ALGORITHMS, plain)
        end
    end

    @testset "phase accounting" begin
        E.resetPhases!()
        @test isempty(E.PHASES)

        # nesting: an inner phase's allocations are also counted by the outer one
        outer = E.withPhase(:total) do
            x = zeros(200_000)
            inner = E.withPhase(:lmo) do
                sum(zeros(400_000))
            end
            sum(x) + inner
        end
        @test outer == 0.0                                   # value passes through
        @test E.PHASES[:total].calls == 1
        @test E.PHASES[:lmo].calls == 1
        @test E.PHASES[:lmo].alloc > 0
        @test E.PHASES[:total].alloc >= E.PHASES[:lmo].alloc  # phases are inclusive

        # repeated entry accumulates
        for _ in 1:3
            E.withPhase(:lmo) do
                sum(zeros(1000))
            end
        end
        @test E.PHASES[:lmo].calls == 4

        # an exception still closes the phase and propagates unchanged
        @test_throws ErrorException E.withPhase(:cp) do
            error("boom")
        end
        @test E.PHASES[:cp].calls == 1

        rep = E.phaseReport()
        @test any(startswith(l, "mem_total_alloc_gib:") for l in rep)
        @test any(startswith(l, "mem_lmo_calls: 4") for l in rep)

        E.resetPhases!()
        @test isempty(E.PHASES)
    end

    @testset "notePhaseModel! records the largest model" begin
        E.resetPhases!()
        m = Model(); @variable(m, y[1:4]); @constraint(m, sum(y) == 1)
        E.notePhaseModel!(:cp, m; nnz = true)
        @test E.PHASES[:cp].nvars == 4
        @test E.PHASES[:cp].ncons >= 1
        @test E.PHASES[:cp].nnz == 4
        big = Model(); @variable(big, z[1:9]); @constraint(big, sum(z) == 1)
        E.notePhaseModel!(:cp, big)
        @test E.PHASES[:cp].nvars == 9          # keeps the max
        E.resetPhases!()
    end
end
