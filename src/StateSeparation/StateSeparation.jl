

function solveModel(optmodel::OptModel, BST, param::Param, silent = false, restrict = false)
   if param.solver == "MSK"
      model = optmodel.model
      setMosekParam(model, param)
      status, solverstatus, primalobj, dualobj = solveMSK(model, param, silent)
      sol = nothing
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
               #push!(Zrcosts, Dict(:RE => reduced_cost.(Zvars[:RE][sys.Zind]), :IM => reduced_cost.(Zvars[:IM][sys.Zind])))
               return nothing
            end
            traverseDPSBST(1, BST, getValueAuxSysVars)
            sol = (Xvals = Xvals, Yval = Yval, Zvals = Zvals, dualobj = dualobj, primalobj = primalobj)
         end
      end
      return status, solverstatus, sol
   end
end



function separate!(problem::Problem, param::Param, effortlevel = 0, globalobbt = false)
   # create a StateSeparator problem data structure
   print("--separating...\n")
   stateseparator = StateSeparator(problem, param)
   BST = problem.BST
   if stateseparator.param.log_level > 0
      #print("stateseparator system dimension: $(stateseparator.problem.dimH), number of subsystems: $(stateseparator.problem.nsubs)\n")
   end
   # establish the initial primal bound
   RunHeuristicsRoot(stateseparator)
   if effortlevel == 0 && stateseparator.primaloutbd > stateseparator.cutoffbound
      print("--found a cut state by heuristic $(stateseparator.primaloutbd) $(stateseparator.cutoffbound) \n")
      return  stateseparator.primaloutbd, stateseparator.dualbd, stateseparator.primalHbar, stateseparator.primalsol
   end
   # initial the node list
   rootnode = creatRootNode(problem.dims, BST, problem.Zdims, param.feas_tol, problem.globalZBs, problem.fixvars)
   stateseparatorAddNode!(stateseparator, rootnode)
   if globalobbt && param.max_obbt > 0
      isfeasible = BoundTighten(stateseparator, rootnode, globalobbt)
      if !isfeasible
         stateseparator.dualbd  = stateseparator.primalbd
         return
      end
      if globalobbt
         problem.globalZBs = rootnode.ZBs
         problem.fixvars = rootnode.fixvars
      end
   end
   if effortlevel == 0 && stateseparator.primaloutbd > stateseparator.cutoffbound
      print("--found a cut state by heuristic $(stateseparator.primalbd) $(stateseparator.cutoffbound) \n")
      return  stateseparator.primaloutbd, stateseparator.dualbd, stateseparator.primalHbar, stateseparator.primalsol
   end
   stateseparator.selectnode = 1
   nnodes = 1
   maxnnodes = effortlevel >= 2 ? param.maxeffortnnodes : param.maxnnodes
   while nnodes <= maxnnodes
      status = RelaxUnsolve
      sol = nothing
      while !isempty(stateseparator.opennodes)
         # solve the full relaxation
         focusnodeid = pop!(stateseparator.opennodes)
         focusnode = stateseparatorGetNode(stateseparator, focusnodeid)
         if focusnode.pruned
            continue
         end
         optmodel = initRelaxationNode(stateseparator, focusnode, stateseparator.primalbd, effortlevel >= 2 && param.enable_dps )
         #if effortlevel >= 2
         #   addCuts(stateseparator, optmodel, focusnode, param)
         #end
         focusnode.localdualbd = Inf
         status, solverstatus, sol = solveModel(optmodel, BST, stateseparator.param)
         if isnothing(status) || !(status == RelaxFeasible || status == RelaxOptimal || status == RelaxInfeasible)
            stateseparator.status = StateSeparatorNotFinished
            print("nonvalid status ", solverstatus, " ", status, "\n")
            return
         end
         # prune by infeasiblity
         if status == RelaxInfeasible
            focusnode.localdualbd = -Inf
            focusnode.pruned = true
            continue
         end
         focusnode.sol = sol
         focusnode.localdualbd = sol.dualobj
         # heuristics
         if status == RelaxFeasible || status == RelaxOptimal
            focusnode.heursol = RunHeuristics(stateseparator, sol)
         end
         # separation from pool
         #if cutrelaxationround <= maxcutrelaxationround && effortlevel >= 2
         #   time_cuts = @elapsed separateCutLocal(stateseparator, optmodel, focusnode, param)
         #   println("Time to add cuts: $(time_cuts) seconds\n")
         #end
      end

      # update the node status and global dual bound of the tree
      stateseparatorUpdateTree!(stateseparator)

      # the whole tree is pruned
      if length(stateseparator.leaves) == 0
         if stateseparator.param.log_level > 0
            print("--no leaves, terminated\n")
         end
         return  stateseparator.primalbd, stateseparator.primalbd, stateseparator.primalHbar, stateseparator.primalsol
      # the whole tree is pruned
      elseif  abs(stateseparator.dualbd - stateseparator.primalbd) < param.obj_tol || abs(stateseparator.dualbd - stateseparator.primalbd) / max(abs(stateseparator.dualbd), abs(stateseparator.primalbd)) < param.rel_obj_tol
         if stateseparator.param.log_level > 0
            print("--gap closed, terminated\n")
         end
         return  stateseparator.primalbd, stateseparator.primalbd, stateseparator.primalHbar, stateseparator.primalsol
      elseif effortlevel < 2 && stateseparator.primaloutbd > stateseparator.cutoffbound
         if stateseparator.param.log_level > 0 && nnodes >= param.minnnodes
            print("--early stop, find a state\n")
         end
         return  stateseparator.primaloutbd, stateseparator.dualbd, stateseparator.primalHbar, stateseparator.primalsol
      elseif effortlevel < 2 && stateseparator.dualbd < stateseparator.cutoffbound
         if stateseparator.param.log_level > 0
            print("--early stop, no good state\n")
         end
         return  stateseparator.primalbd, stateseparator.dualbd, stateseparator.primalHbar, stateseparator.primalsol
      end

      # select node
      branchnode = stateseparatorSelectNode(stateseparator)
      stateseparator.selectnode = branchnode.nodeid
      sol = branchnode.sol
      # branch and create subnodes
      @assert !isnothing(sol)
      bpoint, bscore = executeBranchRules(stateseparator, branchnode, sol)
      if stateseparator.param.log_level > 0
         print("--node number: $(length(stateseparator.nodes)), dual bound: $(stateseparator.dualbd), branchnode bound: $(branchnode.localdualbd),
         primal out bound: $(stateseparator.primaloutbd), primal bound: $(stateseparator.primalbd), cutoff bound: $(stateseparator.cutoffbound),  gap: $(stateseparator.dualbd - stateseparator.primalbd),
         rel_gap: $(abs(stateseparator.dualbd - stateseparator.primalbd) / max(abs(stateseparator.dualbd), abs(stateseparator.primalbd))), nleaves: $(length(stateseparator.leaves)),
         Zdepth: $(bpoint[10]), bbounddistance: $(bpoint[9]), branchscore: $(bscore), maxdepth: $(stateseparator.maxdepth), focusdepth: $(branchnode.depth)\n")
      end
      stateseparatorCreateBranchNodes!(stateseparator, branchnode, bpoint[1], bpoint[2], bpoint[3], bpoint[4], bpoint[5])
      nnodes = stateseparatorGetNNodes(stateseparator)
      if stateseparator.param.log_level > 0 && nnodes == maxnnodes
         print("--max node number reached: $(length(stateseparator.nodes)), terminated\n")
      end
  end
  maxtry = 100
  trycount = 0
  while trycount < maxtry && stateseparator.primaloutbd < stateseparator.cutoffbound
     trycount += 1
     RunHeuristicsRoot(stateseparator)
  end
  return stateseparator.primalbd, stateseparator.dualbd, stateseparator.primalHbar, stateseparator.primalsol
end