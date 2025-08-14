# StateSeparator problem data
mutable struct StateSeparator
   problem::Problem
   param::Param
   opennodes::Vector{Int64}
   leaves::Set{Int}
   maxdepth::Int
   plungedepth::Int
   nodes::Vector{Node}
   dualbd::Float64
   primalbd::Float64
   primaloutbd::Float64
   cutoffbound::Float64
   primalsol
   primalHbar
   selectnode::Int
   seed
   status

   function StateSeparator(problem::Problem, param::Param)
      dualbd = Inf
      primalbd = -Inf
      primaloutbd = -Inf
      cutoffbound = problem.cutoffbound
      opennodes = []
      leaves = Set{Int}()
      maxdepth = 0
      plungedepth = 0
      nodes = []
      primalsol = nothing
      primalHbar = nothing
      selectnode = 1
      seed = MersenneTwister(param.seed)
      status = RelaxUnsolve
      stateseparator = new(problem, param, opennodes, leaves, maxdepth, plungedepth, nodes,
         dualbd, primalbd, primaloutbd, cutoffbound, primalsol, primalHbar, selectnode, seed, status)
      return stateseparator
   end
 end


 function stateseparatorAddNode!(stateseparator::StateSeparator, node::Node)
    push!(stateseparator.nodes, node)
    push!(stateseparator.opennodes, node.nodeid)
    push!(stateseparator.leaves, node.nodeid)
    stateseparator.maxdepth = max(stateseparator.maxdepth, node.depth)
 end

 function stateseparatorGetNNodes(stateseparator::StateSeparator)
    return length(stateseparator.nodes)
 end

 function stateseparatorGetNode(stateseparator::StateSeparator, nodeid::Int64)
    return stateseparator.nodes[nodeid]
 end

 function stateseparatorPruneSubtree!(stateseparator::StateSeparator, nodeid::Int)
    node = stateseparator.nodes[nodeid]
    if nodeHasChilds(node)
       if !stateseparator.nodes[node.childs[1]].pruned
          stateseparatorPruneSubtree!(stateseparator, node.childs[1])
       end
       if !stateseparator.nodes[node.childs[2]].pruned
          stateseparatorPruneSubtree!(stateseparator, node.childs[2])
       end
    end
 end

 function stateseparatorUpdateTree!(stateseparator::StateSeparator)
    # prune the nodes
    while true
       findprune = false
       for node in stateseparator.nodes
          if !node.pruned && node.localdualbd < stateseparator.primalbd - stateseparator.param.obj_tol
             node.pruned = true
             findprune = true
             #print((node.nodeid, length(stateseparator.nodes), node.localdualbd, stateseparator.primalbd + stateseparator.param.obj_tol))
             stateseparatorPruneSubtree!(stateseparator, node.nodeid)
          end
          if !node.pruned && nodeHasChilds(node) && stateseparator.nodes[node.childs[1]].pruned && stateseparator.nodes[node.childs[2]].pruned
             node.pruned = true
             findprune = true
             #print((node.nodeid, length(stateseparator.nodes), node.localdualbd, stateseparator.primalbd + stateseparator.param.obj_tol))
             stateseparatorPruneSubtree!(stateseparator, node.nodeid)
          end
       end
       if !findprune
          break
       end
    end
    # construct the set of (unpruned) leaves
    empty!(stateseparator.leaves)
    for node in stateseparator.nodes
       node.isleave = false
       if !node.pruned
          if !nodeHasChilds(node)
             push!(stateseparator.leaves, node.nodeid)
             node.isleave = true
          elseif stateseparator.nodes[node.childs[1]].pruned && stateseparator.nodes[node.childs[2]].pruned
             push!(stateseparator.leaves, node.nodeid)
             node.isleave = true
          end
       end
    end
    # update global dual bound
    dualbd = -Inf
    for leave in stateseparator.leaves
       node = stateseparatorGetNode(stateseparator, leave)
       dualbd = max(dualbd, node.localdualbd)
    end
    stateseparator.dualbd = dualbd
 end

