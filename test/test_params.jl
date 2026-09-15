using Test, JuMP
import MathOptInterface as MOI
const E = ExactEntanglement

@testset "Param and size presets" begin
    p = Param()
    @test p.solver == "MSK"
    @test p.max_obbt == 0                       # bound tightening off by default
    @test p.loop == 2

    @testset "presets follow the paper's per-size settings" begin
        for (nsubs, tl, r) in ((3, 3600.0, 266), (4, 7200.0, 522), (5, 10800.0, 662))
            q = Param(time_limit = -1.0)
            E.applySizePreset!(q, nsubs, "LD1")
            @test q.time_limit == tl
            @test q.rank_bound == r
            @test q.maxrounds == r               # CP iteration limit tracks r
        end
        # an explicit positive time limit is respected
        q = Param(time_limit = 42.0); E.applySizePreset!(q, 3, "LD1")
        @test q.time_limit == 42.0
        # the LDR rank sweep applies only at m == 5
        q5 = Param(time_limit = -1.0); E.applySizePreset!(q5, 5, "LDR3")
        @test q5.rank_bound == 700
        q3 = Param(time_limit = -1.0); E.applySizePreset!(q3, 3, "LDR3")
        @test q3.rank_bound == 266
    end

    @testset "run clock" begin
        q = Param(time_limit = 100.0)
        q.start_time = time() - 99.0
        @test !E.isTimeLimitExceeded(q)
        @test E.isTimeLimitNearlyReached(q)      # past 90% and not yet in last phase
        E.extendTimeLimit!(q)
        @test q.is_last
        @test q.time_limit > 100.0
        @test !E.isTimeLimitNearlyReached(q)     # only fires once
    end
end

@testset "conic status classification" begin
    cls(status, ps, ds; sense = JuMP.MIN_SENSE, po = 1.0, du = 1.0, tol = 1e-6,
        tag = E.RelaxInfeasible, unk = false) =
        E.classifyStatus(sense, status, ps, ds, po, du;
                         obj_tol = tol, infeasible_tag = tag, unknown_is_nosolution = unk)

    @test cls(OPTIMAL, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT) == E.RelaxOptimal
    @test cls(INFEASIBLE, MOI.NO_SOLUTION, MOI.NO_SOLUTION) == E.RelaxInfeasible
    # a dual infeasibility certificate is reported per caller's tag
    @test cls(SLOW_PROGRESS, MOI.FEASIBLE_POINT, MOI.INFEASIBILITY_CERTIFICATE;
              tag = E.RelaxInfeasibleCertificate) == E.RelaxInfeasibleCertificate
    @test cls(SLOW_PROGRESS, MOI.NO_SOLUTION, MOI.FEASIBLE_POINT) == E.RelaxNoSolution
    # an unknown status only means "no solution" when the caller asks for it
    @test cls(SLOW_PROGRESS, MOI.UNKNOWN_RESULT_STATUS, MOI.FEASIBLE_POINT) != E.RelaxNoSolution
    @test cls(SLOW_PROGRESS, MOI.UNKNOWN_RESULT_STATUS, MOI.FEASIBLE_POINT; unk = true) == E.RelaxNoSolution
    # a primal/dual gap of the wrong sign means the relaxation is infeasible
    @test cls(ITERATION_LIMIT, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT; po = 1.0, du = 2.0) == E.RelaxInfeasible
    @test cls(ITERATION_LIMIT, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT; po = 1.0, du = 1.0) == E.RelaxFeasible
    @test cls(TIME_LIMIT, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT) == E.RelaxError
end
