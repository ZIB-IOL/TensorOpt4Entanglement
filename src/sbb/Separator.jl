# StateSeparator problem data
mutable struct StateSeparator
   problem::Problem
   param::Param
   opennodes::Vector{Int64}
   leaves::Set{Int}
   maxdepth::Int
   nodes::Vector{Node}
   nsolved::Int
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
      # Param is mutable; reject an invalid tolerance again before using it
      # in objective cutoffs, pruning, or the global support-function bound.
      validateBoundTolerances(param)
      dualbd = Inf
      primalbd = -Inf
      primaloutbd = -Inf
      cutoffbound = problem.cutoffbound
      opennodes = []
      leaves = Set{Int}()
      maxdepth = 0
      nodes = []
      primalsol = nothing
      primalHbar = nothing
      selectnode = 1
      seed = MersenneTwister(param.seed)
      status = RelaxUnsolve
      stateseparator = new(problem, param, opennodes, leaves, maxdepth, nodes, 0,
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
    node.pruned = true
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
    for node in stateseparator.nodes
       if !node.pruned && node.localdualbd < stateseparator.primalbd - stateseparator.param.obj_tol
          stateseparatorPruneSubtree!(stateseparator, node.nodeid)
       end
    end
    # Children are appended after their parent, so a reverse pass propagates
    # exhausted subtrees without repeatedly scanning the entire tree.
    for node in Iterators.reverse(stateseparator.nodes)
       if !node.pruned && nodeHasChilds(node) && all(i -> stateseparator.nodes[i].pruned, node.childs)
          node.pruned = true
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
    # The oracle searches only above cutoffbound. If that domain is exhausted,
    # cutoffbound still bounds all excluded states; -Inf is not a valid bound
    # for the original support function.
    stateseparator.dualbd = max(dualbd, stateseparator.cutoffbound,
        stateseparator.primalbd + stateseparator.param.obj_tol)
 end
