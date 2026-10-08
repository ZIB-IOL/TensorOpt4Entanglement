# ---------------------------------------------------------------------------
# Solver status codes, algorithm parameters, and the run clock.
# ---------------------------------------------------------------------------

@enum Status RelaxUnsolve RelaxOptimal RelaxFeasible RelaxNoSolution RelaxInfeasible RelaxInfeasibleCertificate RelaxError StateSeparatorStart StateSeparatorNotFinished StateSeparatorTerminate

# Part/bound tags used as Dict keys throughout: :RE/:IM select the real or
# imaginary part, :L/:U the lower or upper bound. They are written as plain
# symbol literals everywhere -- do NOT reintroduce aliases such as
# `IM = Symbol("Imaginary")`, which silently fail to match `:IM` keys.

"""
    Param(; kwargs...)

Algorithm parameters. Paper symbols in brackets.

Solver / limits
  `solver`          conic solver tag, currently only `"MSK"` (Mosek)
  `time_limit`      global wall-clock budget in seconds
  `thread`          solver threads
  `log_level`       0 silent … 3 verbose

Tolerances
  `obj_tol`         duality gap for sBB subproblems
  `master_obj_tol`  duality gap for the cutting-plane master
  `tol`, `feas_tol` general and feasibility tolerances

Cutting plane (CP) and active set
  `maxrounds`       CP iteration limit; inside IR it is set to r
  `pointsize_bound` size of the initial active set `|P_1|`
  `rank_bound`      factorisation size [r]; also caps `|P_k|`
  `lazification`    keep separated states in a pool and re-add them lazily
  `pool_size`       pool capacity (`-1` = unbounded)
  `loop`            IR iteration budget; negative values are run-mode
                    sentinels, see `refinementBudget`
  `cp_real_master`  real witness and conjugate-pair columns for real targets
  `cp_rounds_per_ir` optional CP round cap in intermediate IR passes (-1 = off)
  `cp_certify_every` stabilise the witness and certify at the root every N
                     intermediate rounds (0 = off); also stabilise witnesses
                     in the final phase and the witness returned to IR
  `ir_refit_scalar` refit the mixing scalar after trimming the CP support

sBB linear-minimisation oracle
  `maxnnodes`       node limit for ordinary CP iterations
  `maxeffortnnodes` node limit for gap-closing iterations
  `heur_sbb_maxiter` coordinate updates per eigenvector heuristic start
  `heur_sbb_restarts` random starts before solving the root relaxation
  `heur_sbb_node_restarts` starts guided by each node's relaxed local states
  `minnnodes`       minimum nodes before an early stop is reported
  `max_obbt`        rounds of optimisation-based bound tightening (0 = off)
  `relaxation`      which convex relaxation the sBB oracle uses:
                    `:ddps`     the DDPS outer approximation alone -- one PSD,
                                unit-trace Z per tree node plus partial-trace
                                consistency with its children
                    `:ddpsplus` (default) DDPS strengthened with the scalar
                                complex McCormick inequalities
  `is_last`         set once the run enters its gap-closing phase
  `tratio`          fraction of `time_limit` reserved for that phase

LADMM (lifted ADMM)
  `heur_LADMM_maxiter`   outer iterations, warm-started calls
  `heur_LADMM1_maxiter`  outer iterations, first call
  `heur_LADMM_rho`       initial penalty parameter [ζ]
  `heur_LADMM_obj_tol`, `heur_LADMM_step_tol`, `heur_LADMM_gd_tol`
                         objective / step / gradient tolerances
  `heur_MANOPT_maxiter`  quasi-Newton iterations, warm-started calls
  `heur_MANOPT1_maxiter` quasi-Newton iterations, first call

Alternating SDP (Alt-SDP)
  `heur_alternate_iter`    iterations, warm-started calls (`-1` = unlimited)
  `heur_alternate1_iter`   iterations, first call
  `heur_alternate_maxfail` consecutive non-improving iterations tolerated

Misc
  `seed`        RNG seed
  `start_time`  run start, set at construction
  `pdgr`        PDGR.Options with settings specific to the PDGR baseline
"""
mutable struct Param
   solver::String
   time_limit::Float64
   obj_tol::Float64
   master_obj_tol::Float64
   tol::Float64
   feas_tol::Float64
   log_level::Int
   thread::Int
   maxnnodes::Int
   maxeffortnnodes::Int
   minnnodes::Int
   heur_sbb_maxiter::Int
   heur_sbb_restarts::Int
   heur_sbb_node_restarts::Int
   maxrounds::Int
   seed::Int
   heur_LADMM1_maxiter::Int
   heur_LADMM_maxiter::Int
   heur_LADMM_obj_tol::Float64
   heur_LADMM_step_tol::Float64
   heur_LADMM_gd_tol::Float64
   heur_LADMM_rho::Float64
   heur_LADMM_penalty_update::Symbol
   heur_LADMM_conjugates::Bool
   heur_MANOPT_maxiter::Int
   heur_MANOPT1_maxiter::Int
   heur_alternate_iter::Int
   heur_alternate1_iter::Int
   heur_alternate_maxfail::Int
   max_obbt::Int
   relaxation::Symbol
   loop::Int
   start_time::Float64
   is_last::Bool
   tratio::Float64
   lazification::Bool
   pool_size::Int
   pointsize_bound::Int
   rank_bound::Int
   cp_real_master::Bool
   cp_rounds_per_ir::Int
   cp_certify_every::Int
   ir_refit_scalar::Bool
   pdgr::PDGR.Options

   function Param(;
         solver::String = "MSK",
         time_limit::Float64 = 200.0,
         obj_tol::Float64 = 1e-6,
         master_obj_tol::Float64 = 1e-6,
         tol::Float64 = 1e-6,
         feas_tol::Float64 = 1e-6,
         log_level::Int = 1,
         thread::Int = 1,
         maxnnodes::Int = 100,
         maxeffortnnodes::Int = 200,
         minnnodes::Int = 0,
         heur_sbb_maxiter::Int = 100,
         heur_sbb_restarts::Int = 1,
         heur_sbb_node_restarts::Int = 4,
         maxrounds::Int = 100,
         seed::Int = 12345,
         heur_LADMM1_maxiter::Int = 5,
         heur_LADMM_maxiter::Int = 3,
         heur_LADMM_obj_tol::Float64 = 1e-5,
         heur_LADMM_step_tol::Float64 = 1e-4,
         heur_LADMM_gd_tol::Float64 = 1e-4,
         heur_LADMM_rho::Float64 = 1.0,
         heur_LADMM_penalty_update::Symbol = :legacy,
         heur_LADMM_conjugates::Bool = false,
         heur_MANOPT_maxiter::Int = 150,
         heur_MANOPT1_maxiter::Int = 150,
         heur_alternate_iter::Int = 10,
         heur_alternate1_iter::Int = 20,
         heur_alternate_maxfail::Int = 4,
         max_obbt::Int = 0,
         relaxation::Symbol = :ddpsplus,
         loop::Int = 2,
         start_time::Float64 = time(),
         is_last::Bool = false,
         tratio::Float64 = 0.1,
         lazification::Bool = false,
         pool_size::Int = 5000,
         pointsize_bound::Int = 100,
         rank_bound::Int = 500,
         cp_real_master::Bool = false,
         cp_rounds_per_ir::Int = -1,
         cp_certify_every::Int = 0,
         ir_refit_scalar::Bool = false,
         pdgr::PDGR.Options = PDGR.Options())
      heur_LADMM_penalty_update in (:balance, :legacy) ||
         throw(ArgumentError("LADMM penalty update must be :balance or :legacy"))
      (cp_rounds_per_ir == -1 || cp_rounds_per_ir > 0) || throw(ArgumentError("CP rounds per IR pass must be -1 or positive"))
      cp_certify_every >= 0 || throw(ArgumentError("CP certification interval must be nonnegative"))
      heur_sbb_maxiter > 0 || throw(ArgumentError("sBB heuristic iterations must be positive"))
      heur_sbb_restarts > 0 || throw(ArgumentError("sBB heuristic restarts must be positive"))
      heur_sbb_node_restarts > 0 || throw(ArgumentError("sBB node heuristic restarts must be positive"))
      new(solver, time_limit, obj_tol, master_obj_tol, tol, feas_tol, log_level,
          thread, maxnnodes, maxeffortnnodes, minnnodes, heur_sbb_maxiter,
          heur_sbb_restarts, heur_sbb_node_restarts, maxrounds, seed,
          heur_LADMM1_maxiter, heur_LADMM_maxiter, heur_LADMM_obj_tol,
          heur_LADMM_step_tol, heur_LADMM_gd_tol, heur_LADMM_rho,
          heur_LADMM_penalty_update, heur_LADMM_conjugates,
          heur_MANOPT_maxiter, heur_MANOPT1_maxiter,
          heur_alternate_iter,
          heur_alternate1_iter, heur_alternate_maxfail, max_obbt, relaxation,
          loop, start_time, is_last, tratio, lazification, pool_size,
          pointsize_bound, rank_bound, cp_real_master, cp_rounds_per_ir,
          cp_certify_every, ir_refit_scalar, pdgr)
   end
end

"""
    elapsedTime(param)

Seconds since the run started.
"""
elapsedTime(param::Param) = time() - param.start_time

"Seconds left in the shared solver budget; model setup also consumes this budget."
remainingTime(param::Param) = max(param.time_limit - elapsedTime(param), 0.0)

"""
    isTimeLimitExceeded(param)

Whether the global wall-clock budget is used up.
"""
function isTimeLimitExceeded(param::Param)
   elapsed = elapsedTime(param)
   param.log_level > 1 && println("Elapsed time: $elapsed s, limit: $(param.time_limit) s")
   return elapsed >= param.time_limit
end

"""
    isTimeLimitNearlyReached(param)

Whether the run has passed `1 - tratio` of its budget and has not yet switched
to the gap-closing phase. Fires at most once per run: [`extendTimeLimit!`](@ref)
sets `is_last`, after which this returns `false`.
"""
function isTimeLimitNearlyReached(param::Param)
   elapsed = elapsedTime(param)
   param.log_level > 1 && println("Elapsed time: $elapsed s, limit: $(param.time_limit) s, is_last=$(param.is_last)")
   return !param.is_last && elapsed > (1 - param.tratio) * param.time_limit
end

"""
    extendTimeLimit!(param)

Enter the gap-closing phase without extending the requested wall-clock budget.
The historical function name is retained for callers.
"""
function extendTimeLimit!(param::Param)
   param.log_level > 0 &&
      println("entering gap-closing phase: $(remainingTime(param)) s remain")
   param.is_last = true
   return param
end
