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

sBB linear-minimisation oracle
  `maxnnodes`       node limit for ordinary CP iterations
  `maxeffortnnodes` node limit for gap-closing iterations
  `minnnodes`       minimum nodes before an early stop is reported
  `max_obbt`        rounds of optimisation-based bound tightening (0 = off)
  `relaxation`      which convex relaxation the sBB oracle uses:
                    `:ddps`     the DDPS outer approximation alone -- one PSD,
                                unit-trace Z per tree node plus partial-trace
                                consistency with its children
                    `:ddpsplus` (default) DDPS strengthened with the tensor and
                                scalar complex McCormick inequalities
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
   maxrounds::Int
   seed::Int
   heur_LADMM1_maxiter::Int
   heur_LADMM_maxiter::Int
   heur_LADMM_obj_tol::Float64
   heur_LADMM_step_tol::Float64
   heur_LADMM_gd_tol::Float64
   heur_LADMM_rho::Float64
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
         maxrounds::Int = 100,
         seed::Int = 12345,
         heur_LADMM1_maxiter::Int = 5,
         heur_LADMM_maxiter::Int = 3,
         heur_LADMM_obj_tol::Float64 = 1e-5,
         heur_LADMM_step_tol::Float64 = 1e-4,
         heur_LADMM_gd_tol::Float64 = 1e-4,
         heur_LADMM_rho::Float64 = 1.0,
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
         rank_bound::Int = 500)
      new(solver, time_limit, obj_tol, master_obj_tol, tol, feas_tol, log_level,
          thread, maxnnodes, maxeffortnnodes, minnnodes, maxrounds, seed,
          heur_LADMM1_maxiter, heur_LADMM_maxiter, heur_LADMM_obj_tol,
          heur_LADMM_step_tol, heur_LADMM_gd_tol, heur_LADMM_rho,
          heur_MANOPT_maxiter, heur_MANOPT1_maxiter, heur_alternate_iter,
          heur_alternate1_iter, heur_alternate_maxfail, max_obbt, relaxation,
          loop, start_time, is_last, tratio, lazification, pool_size,
          pointsize_bound, rank_bound)
   end
end

"""
    elapsedTime(param)

Seconds since the run started.
"""
elapsedTime(param::Param) = time() - param.start_time

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

Enter the gap-closing phase: mark the run as `is_last` and give back the time
already spent beyond the `1 - tratio` mark, so the final sBB calls get the full
`tratio` share of the budget.
"""
function extendTimeLimit!(param::Param)
   elapsed = elapsedTime(param)
   extra = max(elapsed - (1 - param.tratio) * param.time_limit, 0)
   param.log_level > 0 &&
      println("entering gap-closing phase: time limit $(param.time_limit) -> $(param.time_limit + extra) s")
   param.is_last = true
   param.time_limit += extra
   return param
end

