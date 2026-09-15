# ---------------------------------------------------------------------------
# Mosek configuration and conic-solve result classification.
# ---------------------------------------------------------------------------

function setMosekParam(model::Model, param::Param, isdualsimplex = false)
   set_optimizer(model, Mosek.Optimizer)
   set_attribute(model, "MSK_IPAR_NUM_THREADS", param.thread)
   set_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", param.time_limit)
   if isdualsimplex
      #set_attribute(model, "MSK_IPAR_OPTIMIZER", Mosek.MSK_OPTIMIZER_DUAL_SIMPLEX)
   end
   # 4 interior point, 2 automatic, 1 dual simplex
end

function classifyStatus(sense, status, primal_status, dual_status, primalobj, dualobj;
                        obj_tol, infeasible_tag, unknown_is_nosolution)
   if status == OPTIMAL
      return RelaxOptimal
   elseif status == INFEASIBLE || dual_status == INFEASIBILITY_CERTIFICATE
      return infeasible_tag
   elseif status == SLOW_PROGRESS || status == ITERATION_LIMIT
      nosolution = primal_status == NO_SOLUTION ||
         (unknown_is_nosolution &&
          (primal_status == UNKNOWN_RESULT_STATUS || dual_status == UNKNOWN_RESULT_STATUS))
      if nosolution
         return RelaxNoSolution
      elseif sense == JuMP.MIN_SENSE && primalobj + obj_tol < dualobj
         return RelaxInfeasible
      elseif sense == JuMP.MAX_SENSE && primalobj > dualobj + obj_tol
         return RelaxInfeasible
      else
         return RelaxFeasible
      end
   end
   return RelaxError
end

"""
    solveMSK(model, param, silent = false)

Optimise `model` and classify the outcome.
Returns `(relaxstatus, solver_status, primalobj, dualobj)`.
"""
function solveMSK(model::Model, param::Param, silent = false)
   if param.log_level <= 2 || silent
      set_silent(model)
   end
   optimize!(model)
   status = termination_status(model)
   primalobj = objective_value(model)
   dualobj = dual_objective_value(model)
   relaxstatus = classifyStatus(objective_sense(model), status, JuMP.primal_status(model), JuMP.dual_status(model),
                                primalobj, dualobj;
                                obj_tol = param.obj_tol,
                                infeasible_tag = RelaxInfeasible,
                                unknown_is_nosolution = false)
   relaxstatus == RelaxOptimal && (dualobj = primalobj)

   if param.log_level > 1 && !silent
      param.log_level > 2 && print(solution_summary(model))
      print((status, relaxstatus, primalobj, dualobj), "\n")
   end
   return relaxstatus, status, primalobj, dualobj
end

