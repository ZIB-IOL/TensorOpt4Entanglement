using Test, JuMP
import MathOptInterface as MOI
const E = ExactEntanglement

@testset "Param and size presets" begin
    p = Param()
    @test p.solver == "MSK"
    @test p.max_obbt == 0                       # bound tightening off by default
    @test p.loop == 2
    @test p.heur_sbb_node_restarts == 4

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
        # the rank sweep applies only at m == 5
        q5 = Param(time_limit = -1.0); E.applySizePreset!(q5, 5, "LADMM_700")
        @test q5.rank_bound == 700
        q3 = Param(time_limit = -1.0); E.applySizePreset!(q3, 3, "LADMM_700")
        @test q3.rank_bound == 266
        # a legacy code selects the same r, since the preset resolves aliases
        q5l = Param(time_limit = -1.0); E.applySizePreset!(q5l, 5, "LDR3")
        @test q5l.rank_bound == 700

        for n in (4, 5), algo in ("IR", "IR-nolazy", "IR-DDPS", "LDL")
            q = Param(time_limit=42.0); E.applySizePreset!(q, n, algo)
            @test q.time_limit == 42.0
            @test q.heur_MANOPT_maxiter == q.heur_MANOPT1_maxiter == 40
            @test q.heur_LADMM_maxiter == q.heur_LADMM1_maxiter == 40
            @test q.cp_rounds_per_ir == q.cp_certify_every == 4
            @test q.cp_real_master && q.ir_refit_scalar
            @test q.heur_LADMM_penalty_update == :balance
            E.applyHeuristicOverrides!(q, Dict("cp-real-master"=>false,
                "cp-rounds-per-ir"=>-1, "cp-certify-every"=>0,
                "ir-refit-scalar"=>false, "heur-ladmm-penalty-update"=>"legacy"))
            @test !q.cp_real_master && !q.ir_refit_scalar
            @test q.cp_rounds_per_ir == -1 && q.cp_certify_every == 0
            @test q.heur_LADMM_penalty_update == :legacy
        end
        for (n, algo) in ((3, "IR"), (5, "CP"), (5, "LADMM"))
            q = Param(); E.applySizePreset!(q, n, algo)
            @test q.heur_LADMM_penalty_update == :legacy
            @test q.cp_rounds_per_ir == -1 && q.cp_certify_every == 0
            @test !q.cp_real_master
        end

        # Explicit limits take precedence over the per-size presets.
        q = Param(); E.applySizePreset!(q, 5, "IR")
        E.applyHeuristicOverrides!(q, Dict("heur-manopt-maxiter"=>40,
            "heur-ladmm-maxiter"=>12, "heur-ladmm1-maxiter"=>16))
        @test q.heur_MANOPT_maxiter == q.heur_MANOPT1_maxiter == 40
        @test q.heur_LADMM_maxiter == 12
        @test q.heur_LADMM1_maxiter == 16
        E.applyHeuristicOverrides!(q, Dict("heur-manopt-maxiter"=>nothing))
        @test q.heur_MANOPT_maxiter == 40
        @test_throws ArgumentError E.applyHeuristicOverrides!(q, Dict("heur-manopt-maxiter"=>0))
        E.applyHeuristicOverrides!(q, Dict("maxnnodes"=>7,"maxeffortnnodes"=>15,
            "heur-sbb-restarts"=>4,"heur-sbb-maxiter"=>200,"heur-sbb-node-restarts"=>10))
        @test q.maxnnodes == 7 && q.maxeffortnnodes == 15
        @test q.heur_sbb_restarts == 4 && q.heur_sbb_maxiter == 200
        @test q.heur_sbb_node_restarts == 10
        E.applyHeuristicOverrides!(q, Dict("maxnnodes"=>nothing))
        @test q.maxnnodes == 7
        @test_throws ArgumentError E.applyHeuristicOverrides!(q, Dict("maxnnodes"=>-1))
        @test_throws ArgumentError Param(heur_sbb_restarts=0)
        @test_throws ArgumentError E.applyHeuristicOverrides!(q, Dict("cp-rounds-per-ir"=>0))
        @test_throws ArgumentError E.applyHeuristicOverrides!(q, Dict("cp-certify-every"=>-1))
        @test_throws ArgumentError E.applyHeuristicOverrides!(q, Dict("heur-ladmm-penalty-update"=>"invalid"))
    end

    @testset "run clock" begin
        q = Param(time_limit = 100.0)
        q.start_time = time() - 99.0
        @test !E.isTimeLimitExceeded(q)
        @test E.isTimeLimitNearlyReached(q)      # past 90% and not yet in last phase
        E.extendTimeLimit!(q)
        @test q.is_last
        @test q.time_limit == 100.0
        @test 0 < E.remainingTime(q) < 2
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
    # A wrong-sign gap is a numerical error; it must not trigger infeasibility pruning.
    @test cls(ITERATION_LIMIT, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT; po = 1.0, du = 2.0) == E.RelaxError
    @test cls(ITERATION_LIMIT, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT; po = 1.0, du = 1.0) == E.RelaxFeasible
    @test cls(TIME_LIMIT, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT) == E.RelaxFeasible
    @test cls(TIME_LIMIT, MOI.NO_SOLUTION, MOI.FEASIBLE_POINT) == E.RelaxNoSolution
    @test cls(TIME_LIMIT, MOI.FEASIBLE_POINT, MOI.NO_SOLUTION) == E.RelaxNoSolution
    @test cls(TIME_LIMIT, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT; po=1.0,du=2.0) == E.RelaxError
    @test cls(TIME_LIMIT, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT; po=Inf) == E.RelaxNoSolution
    @test cls(OPTIMAL, MOI.FEASIBLE_POINT, MOI.NO_SOLUTION) == E.RelaxNoSolution
    @test cls(SLOW_PROGRESS, MOI.FEASIBLE_POINT, MOI.FEASIBLE_POINT; du=-Inf) == E.RelaxNoSolution
    @test_throws ArgumentError Param(cp_rounds_per_ir=0)
    @test_throws ArgumentError Param(cp_certify_every=-1)

    @testset "algorithm names follow the paper" begin
        for name in ("Alt-SDP", "LADMM", "CP", "IR", "DPS", "DDPS+")
            @test haskey(E.ALGORITHMS, name)
        end
        for r in (400, 500, 600, 700, 800, 900)
            @test haskey(E.ALGORITHMS, "LADMM_$r")
        end
        # every legacy code resolves to a registered algorithm
        for (old, new) in E.LEGACY_ALIASES
            @test E.resolveAlgorithm(old) == new
            @test haskey(E.ALGORITHMS, new)
        end
        # an unknown code passes through untouched, so the error names it
        @test E.resolveAlgorithm("nosuch") == "nosuch"
    end

    @testset "the three alias tables agree" begin
        # The legacy algorithm codes are listed in three places, because three
        # languages need them: Julia (CLI), bash (--algo filter and resume),
        # and Python (reading published result files). They must not drift.
        root = joinpath(@__DIR__, "..")

        libsh = read(joinpath(root, "scripts", "lib.sh"), String)
        bash_pairs = Dict{String,String}()
        for m in eachmatch(r"\[([A-Za-z0-9_]+)\]=\"([^\"]+)\"", libsh)
            bash_pairs[m.captures[1]] = m.captures[2]
        end
        @test !isempty(bash_pairs)
        @test bash_pairs == E.LEGACY_ALIASES

        common = read(joinpath(root, "scripts", "tables", "common.py"), String)
        py_pairs = Dict{String,String}()
        for m in eachmatch(r"\"([A-Za-z0-9+_-]+)\": \"([A-Za-z0-9+_-]+)\"", common)
            py_pairs[m.captures[1]] = m.captures[2]
        end
        # common.py maps canonical -> legacy, the inverse of LEGACY_ALIASES;
        # the LADMM_* entries are generated in a comprehension, so check the
        # explicit ones only.
        for (canon, legacy) in py_pairs
            haskey(E.LEGACY_ALIASES, legacy) || continue
            @test E.LEGACY_ALIASES[legacy] == canon
        end
        @test length(py_pairs) >= 13
    end
end
