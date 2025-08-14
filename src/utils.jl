@enum Status RelaxUnsolve RelaxOptimal RelaxFeasible RelaxNoSolution RelaxInfeasible RelaxInfeasibleCertificate RelaxError StateSeparatorStart StateSeparatorNotFinished StateSeparatorTerminate
import Mosek

RE = Symbol("Real")
IM = Symbol("Imaginary")
U = Symbol("Upper")
L = Symbol("Lower")

mutable struct Param
   solver::String
   time_limit::Float64 # time limit
   obj_tol::Float64 # duality gap for subproblems
   rel_obj_tol::Float64 # relative duality gap for subproblems
   master_obj_tol::Float64 # duality gap for subproblems
   tol::Float64 # tolerance
   feas_tol::Float64 # tolerance
   log_level::Int # log level
   thread::Int # thread number
   maxnnodes::Int # max number of nodes
   minnnodes::Int # min number of nodes
   maxrounds::Int # max number of separation rounds
   seed::Int # random seed
   heur_FW_maxiters::Int
   heur_AD_depth::Int
   heur_AD_maxiters::Int
   heur_LADMM1_maxiter::Int
   heur_LADMM_maxiter::Int
   heur_LADMM_obj_tol::Float64
   heur_LADMM_step_tol::Float64
   heur_LADMM_gd_tol::Float64
   heur_MANOPT_depth::Int
   heur_MANOPT_maxiter::Int
   alternae_iter::Int
   nalpha::Int
   extrascale::Float64
   inout_maxreset::Int
   effort_freq::Int
   norm::Int
   multcut_varsize::Int
   multcut_initlpsize::Int
   max_obbt::Int
   freq_globalobbt::Int
   enable_dps::Bool
   loop::Int
   start_time::Float64 # time for state separation
   lazification::Bool # whether to use lazification
   pool_size::Int # size of the pool for states
   pointsize_bound::Int # lower bound for rank
   rank_bound::Int # lower bound for rank

   function Param(; solver::String="MSK", time_limit::Float64=200.0, obj_tol::Float64=1e-6, rel_obj_tol::Float64=7e-3, master_obj_tol::Float64=1e-6, tol::Float64 = 1e-6, feas_tol::Float64 = 1e-6, log_level::Int=1,
         thread::Int=1, maxnnodes::Int=100, minnnodes::Int=10, maxrounds::Int=100, seed::Int=12345, heur_FW_maxiters::Int = 20000, heur_AD_depth::Int = 1, heur_AD_maxiters::Int = 300,
         heur_LADMM1_maxiter::Int = 5, heur_LADMM_obj_tol::Float64 = 1e-5, heur_LADMM_step_tol::Float64 = 1e-4, heur_LADMM_gd_tol::Float64 = 1e-4,  heur_LADMM_maxiter::Int = 3, heur_MANOPT_depth::Int = 1, heur_MANOPT_maxiter::Int = 150, alternae_iter::Int= 100, nalpha::Int = 5, extrascale::Float64 = 0.0, inout_maxreset::Int = 3, effort_freq::Int = 1000, norm::Int = 2,
         multcut_varsize::Int = 5, multcut_initlpsize::Int = 30, max_obbt::Int = 0, freq_globalobbt::Int = 20, enable_dps::Bool = true, loop::Int = 2, start_time::Float64 = time(), lazification::Bool = false, pool_size::Int = 5000, pointsize_bound::Int = 100, rank_bound::Int = 500)
         new(solver, time_limit, obj_tol, rel_obj_tol, master_obj_tol, tol, feas_tol, log_level, thread, maxnnodes, minnnodes, maxrounds, seed, heur_FW_maxiters, heur_AD_depth,
            heur_AD_maxiters, heur_LADMM1_maxiter,  heur_LADMM_maxiter, heur_LADMM_obj_tol, heur_LADMM_step_tol, heur_LADMM_gd_tol, heur_MANOPT_depth, heur_MANOPT_maxiter,alternae_iter,  nalpha, extrascale, inout_maxreset, effort_freq, norm, multcut_varsize,
            multcut_initlpsize, max_obbt, freq_globalobbt, enable_dps, loop, start_time, lazification, pool_size, pointsize_bound, rank_bound)
   end
end

function get_elapsed_time(param::Param)
    return time() - param.start_time
end

function get_remaining_time(param::Param)
    elapsed = get_elapsed_time(param)
    return max(0.0, param.time_limit - elapsed)
end

function is_time_limit_exceeded(param::Param)
   time = get_elapsed_time(param)
   println("Elapsed time: $time seconds, Time limit: $(param.time_limit) seconds")
   return time >= param.time_limit
end

function is_time_limit_last(param::Param)
   elapsed = get_elapsed_time(param)
   println("Elapsed time: $elapsed seconds, Time limit: $(param.time_limit) seconds")
   return elapsed >= 0.9 * param.time_limit
end

function reset_timer!(param::Param)
    param.start_time = time()
end

function setMosekParam(model::Model, param::Param, isdualsimplex = false)
   set_optimizer(model, Mosek.Optimizer)
   set_attribute(model, "MSK_IPAR_NUM_THREADS", param.thread)
   set_attribute(model, "MSK_DPAR_OPTIMIZER_MAX_TIME", param.time_limit)
   if isdualsimplex
      #set_attribute(model, "MSK_IPAR_OPTIMIZER", Mosek.MSK_OPTIMIZER_DUAL_SIMPLEX)
   end
   # 4 interior point, 2 automatic, 1 dual simplex
end

function check(substates, dims)
    nrank1 = length(substates)
    nsubs = length(dims)
    for i in 1:nrank1
        substate = substates[i]
        for j in 1:nsubs
            @assert( length(substate[j]) == dims[j])
        end
    end
end


function minimize_quadratic_on_unit_interval(a, b, c)
    if abs(a) < 1e-12
        zstar = b > 0 ? 0.0 : 1.0
    else
        zstar = -b / (2a)
        zstar = clamp(zstar, 0.0, 1.0)
    end
    f0 = c
    f1 = a + b + c
    fz = a*zstar^2 + b*zstar + c
    if fz <= f0 && fz <= f1
        return zstar, fz
    elseif f0 <= f1
        return 0.0, f0
    else
        return 1.0, f1
    end
end

function realinner(Mdir_c, Min_c)
   inner_real = dot(real(Mdir_c), real(Min_c)) + dot(imag(Mdir_c), imag(Min_c))
   return inner_real
end

function solveMSK(model::Model, param::Param, silent = false)

   if param.log_level <= 2 || silent
      set_silent(model)
   end
   optimize!(model)
   status = termination_status(model)
   primal_status = JuMP.primal_status(model)
   dual_status = JuMP.dual_status(model)
   dualobj = dual_objective_value(model)
   primalobj = objective_value(model)
   relaxstatus = RelaxError
   if status == OPTIMAL
      relaxstatus = RelaxOptimal
      dualobj = primalobj
   elseif status == INFEASIBLE || dual_status == INFEASIBILITY_CERTIFICATE
      relaxstatus = RelaxInfeasible
   elseif status == SLOW_PROGRESS || status == ITERATION_LIMIT
      if primal_status == NO_SOLUTION
         relaxstatus = RelaxNoSolution
      elseif objective_sense(model) == JuMP.MIN_SENSE && primalobj + param.obj_tol  < dualobj
         relaxstatus = RelaxInfeasible
      elseif objective_sense(model) == JuMP.MAX_SENSE && primalobj > dualobj + param.obj_tol
         relaxstatus = RelaxInfeasible
      else
         relaxstatus = RelaxFeasible
      end
   end

   if param.log_level > 1 && !silent
      if param.log_level > 2
         print(solution_summary(model))
      end
      print((status, relaxstatus, primalobj, dualobj), "\n")
   end
   return relaxstatus, status, primalobj, dualobj
end

function cumulativeAdd!(c::Vector{T}) where T
    for i in 2:length(c)
        c[i] += c[i - 1]  # Add the previous element to the current element
    end
    return c
end

function unravel_index(k, dims)
   idxs = []
   for d in reverse(dims)
      pushfirst!(idxs, mod(k, d)+1)
      k ÷= d
   end
   return idxs
end


function get_indexmap(dims)
   D = prod(dims)
   indexmap = Matrix{Vector{Tuple{Int, Int, Int}}}(undef, D, D)
   for row in 1:D, col in 1:D
       indexmap[row, col] = Tuple{Int, Int, Int}[]
   end
   cdims = cumulativeAdd!(deepcopy(dims))
   for i in 0:D-1, j in 0:D-1
      i_idx = unravel_index(i, dims)
      j_idx = unravel_index(j, dims)
      for (k, i_) in enumerate(i_idx)
         dim = dims[k]
         idx_start = (cdims[k] - dim) * 2
         idx_real = idx_start + i_
         idx_imag = idx_start + dim + i_
         push!(indexmap[i+1,j+1], (idx_real, idx_imag, 1) )
      end
      for (k, j_) in enumerate(j_idx)
         dim = dims[k]
         idx_start = (cdims[k] - dim) * 2
         idx_real = idx_start + j_
         idx_imag = idx_start + dim + j_
         push!(indexmap[i+1,j+1], (idx_real, idx_imag, -1) )
      end
   end
   return indexmap
end
