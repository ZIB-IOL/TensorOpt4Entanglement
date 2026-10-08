

function solveModel(optmodel::OptModel, BST, param::Param, silent = false, restrict = false)
   if param.solver == "MSK"
      model = optmodel.model
      setMosekParam(model, param)
      status, solverstatus, primalobj, dualobj = solveMSK(model, param, silent)
      sol = nothing
      if status in (RelaxOptimal, RelaxFeasible) && !(has_values(model) && has_duals(model))
         status = RelaxNoSolution
      end
      if status == RelaxOptimal || status == RelaxFeasible
         if restrict
            Xval = Dict(:RE => value.(optmodel.Xs[:RE]), :IM => value.(optmodel.Xs[:IM]))
            sol = (Xval = Xval, dualobj = dualobj, primalobj = primalobj)
         else
            Xvals = Dict(:RE => [value.(X) for X in optmodel.Xs[:RE]], :IM => [value.(X) for X in optmodel.Xs[:IM]])
            Yval = Dict(:RE => value.(optmodel.Y[:RE]), :IM => value.(optmodel.Y[:IM]))
            Zvals = []
            Zs = optmodel.Zs
            function getValueAuxSysVars(sys, leftsys, rightsys, retleft, retright)
               push!(Zvals, Dict(:RE => value.(Zs[:RE][sys.Zind]), :IM => value.(Zs[:IM][sys.Zind])))
               return nothing
            end
            traverseDPSBST(1, BST, getValueAuxSysVars)
            sol = (Xvals = Xvals, Yval = Yval, Zvals = Zvals, dualobj = dualobj, primalobj = primalobj)
         end
      end
      return status, solverstatus, sol
   end
end

"""
    threshold!(problem, param, effortlevel = 0)

Solve the tensor-RLT / DDPS+ relaxation once at the sBB root and return its
objective: a valid lower bound on the white-noise mixing threshold (`-a RLT`).
"""
function threshold_!(problem::Problem, param::Param, effortlevel = 0)
   remainingTime(param) <= 0 && return 0.0
   # create a StateSeparator problem data structure
   print("--separating...\n")
   stateseparator = StateSeparator(problem, param)
   BST = problem.BST

   # initial the node list
   rootnode = createRootNode(problem.dims, BST, problem.Zdims, param.feas_tol, problem.globalZBs, problem.fixvars)
   stateseparatorAddNode!(stateseparator, rootnode)
   stateseparator.selectnode = 1

   focusnodeid = pop!(stateseparator.opennodes)
   focusnode = stateseparatorGetNode(stateseparator, focusnodeid)
   optmodel = initRelaxationThreshold(stateseparator, focusnode)
   notePhaseModel!(:lmo, optmodel.model; nnz = true)

   focusnode.localdualbd = Inf
   status, solverstatus, sol = solveModel(optmodel, BST, stateseparator.param)
   if isnothing(status) || !(status == RelaxFeasible || status == RelaxOptimal || status == RelaxInfeasible)
      stateseparator.status = StateSeparatorNotFinished
      print("nonvalid status ", solverstatus, " ", status, "\n")
      # The physical bound zero remains valid if the root solve has no result.
      return 0.0
   end
   isnothing(sol) && return 0.0
   return max(0.0, sol.dualobj)
end

function separationResult(ss::StateSeparator)
   return ss.primaloutbd, ss.dualbd, ss.primalHbar, ss.primalsol
end

"""
    separate!(problem, param, effortlevel = 0, globalobbt = false)

The linear-minimisation oracle: a spatial branch-and-bound over the relaxations
of `StateRelaxation.jl`, looking for an extreme point of the separable tensor
cone that violates the current master cut.

Returns `(primal, dual, Hbar, sol)`. `dual` bounds the maximum witness value
over product states from above, including when the node limit interrupts the
search. CP uses this upper bound to obtain its threshold lower bound.

`effortlevel` 0/1 stop as soon as a violated cut is found; level 2 is the
gap-closing mode used once `param.is_last` is set, and runs to the larger
`param.maxeffortnnodes` node limit.
"""
function separate_!(problem::Problem, param::Param, effortlevel = 0, globalobbt = false)
   # create a StateSeparator problem data structure
   print("--separating...\n")
   stateseparator = StateSeparator(problem, param)
   BST = problem.BST
   if stateseparator.param.log_level > 0
   end
   # establish the initial primal bound
   RunHeuristicsRoot(stateseparator)
   remainingTime(param) <= 0 && return separationResult(stateseparator)
   if effortlevel == 0 && stateseparator.primaloutbd > stateseparator.cutoffbound
      print("--found a cut state by heuristic $(stateseparator.primaloutbd) $(stateseparator.cutoffbound) \n")
      return separationResult(stateseparator)
   end
   # initial the node list
   rootnode = createRootNode(problem.dims, BST, problem.Zdims, param.feas_tol, problem.globalZBs, problem.fixvars)
   stateseparatorAddNode!(stateseparator, rootnode)
   if globalobbt && param.max_obbt > 0
      isfeasible = BoundTighten(stateseparator, rootnode, globalobbt)
      if !isfeasible
         rootnode.pruned = true
         stateseparatorUpdateTree!(stateseparator)
         return separationResult(stateseparator)
      end
      if globalobbt
         problem.globalZBs = rootnode.ZBs
         problem.fixvars = rootnode.fixvars
      end
   end
   if effortlevel == 0 && stateseparator.primaloutbd > stateseparator.cutoffbound
      print("--found a cut state by heuristic $(stateseparator.primalbd) $(stateseparator.cutoffbound) \n")
      return separationResult(stateseparator)
   end
   stateseparator.selectnode = 1
   maxnnodes = effortlevel >= 2 ? param.maxeffortnnodes : param.maxnnodes
   while stateseparator.nsolved < maxnnodes && remainingTime(param) > 0
      status = RelaxUnsolve
      sol = nothing
      while !isempty(stateseparator.opennodes) && stateseparator.nsolved < maxnnodes
         if remainingTime(param) <= 0
            stateseparatorUpdateTree!(stateseparator)
            return separationResult(stateseparator)
         end
         # solve the full relaxation
         focusnodeid = pop!(stateseparator.opennodes)
         focusnode = stateseparatorGetNode(stateseparator, focusnodeid)
         if focusnode.pruned
            continue
         end
         optmodel = initRelaxationNode(stateseparator, focusnode, stateseparator.primalbd)
         notePhaseModel!(:lmo, optmodel.model; nnz = true)
         stateseparator.nsolved += 1
         status, solverstatus, sol = solveModel(optmodel, BST, stateseparator.param)
         if isnothing(status) || !(status == RelaxFeasible || status == RelaxOptimal || status == RelaxInfeasible)
            stateseparator.status = StateSeparatorNotFinished
            print("nonvalid status ", solverstatus, " ", status, "\n")
            stateseparatorUpdateTree!(stateseparator)
            return separationResult(stateseparator)
         end
         # prune by infeasiblity
         if status == RelaxInfeasible
            focusnode.localdualbd = -Inf
            focusnode.pruned = true
            continue
         end
         focusnode.sol = sol
         # A child's domain is contained in its parent's. Both dual bounds
         # remain valid, including if a time-limited solve gives a weaker one.
         focusnode.localdualbd = min(focusnode.localdualbd, sol.dualobj)
         # heuristics
         if status == RelaxFeasible || status == RelaxOptimal
            focusnode.heursol = RunHeuristics(stateseparator, sol)
         end
         # separation from pool
      end

      # update the node status and global dual bound of the tree
      stateseparatorUpdateTree!(stateseparator)

      # the whole tree is pruned
      if length(stateseparator.leaves) == 0
         if stateseparator.param.log_level > 0
            print("--no leaves, terminated\n")
         end
         return separationResult(stateseparator)
      # the whole tree is pruned
      elseif stateseparator.dualbd <= stateseparator.primalbd + param.obj_tol
         if stateseparator.param.log_level > 0
            print("--gap closed, terminated dualbd: $(stateseparator.dualbd) primalbd: $(stateseparator.primalbd)  \n")
         end
         return separationResult(stateseparator)
      elseif effortlevel < 2 && stateseparator.primaloutbd > stateseparator.cutoffbound
            if stateseparator.param.log_level > 0 && stateseparator.nsolved >= param.minnnodes
            print("--early stop, find a state\n")
         end
         return separationResult(stateseparator)
      elseif effortlevel < 2 && stateseparator.dualbd < stateseparator.cutoffbound
         if stateseparator.param.log_level > 0
            print("--early stop, no good state\n")
         end
         return separationResult(stateseparator)
      end

      if stateseparator.nsolved >= maxnnodes
         param.log_level > 0 && println("--node solve limit reached: $(stateseparator.nsolved), dual bound: $(stateseparator.dualbd), primal bound: $(stateseparator.primalbd)")
         break
      end

      # select node
      branchnode = stateseparatorSelectNode(stateseparator)
      stateseparator.selectnode = branchnode.nodeid
      sol = branchnode.sol
      # branch and create subnodes
      @assert !isnothing(sol)
      bpoint, bscore = executeBranchRules(stateseparator, branchnode, sol)
      # No unfixed interval remains wide enough for a meaningful split. Keep
      # the relaxation bound; absence of a branch is not a gap certificate.
      isnothing(bpoint) && break
      if stateseparator.param.log_level > 0
         print("--node number: $(length(stateseparator.nodes)), dual bound: $(stateseparator.dualbd), branchnode bound: $(branchnode.localdualbd),
         primal out bound: $(stateseparator.primaloutbd), primal bound: $(stateseparator.primalbd), cutoff bound: $(stateseparator.cutoffbound),  gap: $(stateseparator.dualbd - stateseparator.primalbd),
         rel_gap: $(abs(stateseparator.dualbd - stateseparator.primalbd) / max(abs(stateseparator.dualbd), abs(stateseparator.primalbd))), nleaves: $(length(stateseparator.leaves)),
         Zdepth: $(bpoint[10]), bbounddistance: $(bpoint[9]), branchscore: $(bscore), maxdepth: $(stateseparator.maxdepth), focusdepth: $(branchnode.depth)\n")
      end
      stateseparatorCreateBranchNodes!(stateseparator, branchnode, bpoint[1], bpoint[2], bpoint[3], bpoint[4], bpoint[5])
  end
  maxtry = 100
  trycount = 0
  while trycount < maxtry && stateseparator.primaloutbd < stateseparator.cutoffbound && remainingTime(param) > 0
     trycount += 1
     RunHeuristicsRoot(stateseparator)
  end
  !isempty(stateseparator.nodes) && stateseparatorUpdateTree!(stateseparator)
  return separationResult(stateseparator)
end
# The two oracle entry points are wrapped so their memory is attributed to the
# :lmo phase. The wrappers return exactly what the implementations return and
# re-raise unchanged, so behaviour is untouched.
separate!(problem::Problem, param::Param, effortlevel = 0, globalobbt = false) =
    withPhase(:lmo) do
        separate_!(problem, param, effortlevel, globalobbt)
    end

threshold!(problem::Problem, param::Param, effortlevel = 0) =
    withPhase(:lmo) do
        threshold_!(problem, param, effortlevel)
    end
