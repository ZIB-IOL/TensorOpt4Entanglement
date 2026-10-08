# ---------------------------------------------------------------------------
# Mosek configuration and conic-solve result classification.
# ---------------------------------------------------------------------------

function setMosekParam(model::Model, param::Param, isdualsimplex = false)
   set_optimizer(model, Mosek.Optimizer)
   set_attribute(model, "MSK_IPAR_NUM_THREADS", param.thread)
   set_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", max(remainingTime(param), 1e-3))
   if isdualsimplex
      #set_attribute(model, "MSK_IPAR_OPTIMIZER", Mosek.MSK_OPTIMIZER_DUAL_SIMPLEX)
   end
   # 4 interior point, 2 automatic, 1 dual simplex
end

function classifyStatus(sense, status, primal_status, dual_status, primalobj, dualobj;
                        obj_tol, infeasible_tag, unknown_is_nosolution)
   if status == INFEASIBLE || dual_status == INFEASIBILITY_CERTIFICATE
      return infeasible_tag
   elseif status in (OPTIMAL, SLOW_PROGRESS, ITERATION_LIMIT, TIME_LIMIT)
      nosolution = primal_status == NO_SOLUTION || dual_status == NO_SOLUTION ||
         !isfinite(primalobj) || !isfinite(dualobj) ||
         (unknown_is_nosolution &&
          (primal_status == UNKNOWN_RESULT_STATUS || dual_status == UNKNOWN_RESULT_STATUS))
      if status == TIME_LIMIT
         feasible = (MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT)
         nosolution |= !(primal_status in feasible && dual_status in feasible)
      end
      if nosolution
         return RelaxNoSolution
      elseif sense == JuMP.MIN_SENSE && primalobj + obj_tol < dualobj
         return RelaxError
      elseif sense == JuMP.MAX_SENSE && primalobj > dualobj + obj_tol
         return RelaxError
      else
         return status == OPTIMAL ? RelaxOptimal : RelaxFeasible
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
   sense = objective_sense(model)
   missingprimal = sense == JuMP.MIN_SENSE ? Inf : -Inf
   missingdual = -missingprimal
   remainingTime(param) <= 0 && return RelaxNoSolution, TIME_LIMIT, missingprimal, missingdual
   set_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", max(remainingTime(param), 1e-3))
   if param.log_level <= 2 || silent
      set_silent(model)
   end
   optimize!(model)
   status = termination_status(model)
   if status == INFEASIBLE || JuMP.dual_status(model) == INFEASIBILITY_CERTIFICATE
      return RelaxInfeasible, status, missingprimal, missingdual
   end
   primalobj = has_values(model) ? objective_value(model) : missingprimal
   dualobj = has_duals(model) ? dual_objective_value(model) : missingdual
   relaxstatus = classifyStatus(objective_sense(model), status, JuMP.primal_status(model), JuMP.dual_status(model),
                                primalobj, dualobj;
                                obj_tol = param.obj_tol,
                                infeasible_tag = RelaxInfeasible,
                                unknown_is_nosolution = false)
   if param.log_level > 1 && !silent
      param.log_level > 2 && print(solution_summary(model))
      print((status, relaxstatus, primalobj, dualobj), "\n")
   end
   return relaxstatus, status, primalobj, dualobj
end
