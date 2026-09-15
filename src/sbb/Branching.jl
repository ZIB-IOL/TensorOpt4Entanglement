function branchViolation(vars, bounds, zval, zrcosts = nothing)
    lowers = Dict((:RE,:RE) => 0., (:IM,:IM) => 0.,
    (:RE,:IM) => 0., (:IM, :RE) => 0.)
    uppers = Dict((:RE,:RE) => 0., (:IM,:IM) => 0.,
    (:RE,:IM) => 0., (:IM, :RE) => 0.)
    zestimates = Dict((:RE,:L) => 0., (:IM,:L) => 0.,
    (:RE,:U) => 0., (:IM, :U) => 0.)
    # compute affine lower and upper estimators for each product of parts of variables
    for var1part in (:RE,:IM)
        for var2part in (:RE,:IM)
            lowers[var1part, var2part] = maximum(affine(vars[var1part][1], vars[var2part][2], bounds[var1part,:L][1], bounds[var2part,:L][2], bounds[var1part,:U][1], bounds[var2part,:U][2]))
            uppers[var1part, var2part] = minimum(affine(vars[var1part][1], vars[var2part][2], bounds[var1part,:U][1], bounds[var2part,:L][2], bounds[var1part,:L][1], bounds[var2part,:U][2]))
        end
    end
    zestimates[:RE,:L] = lowers[:RE, :RE] - uppers[:IM, :IM]
    zestimates[:RE,:U] = uppers[:RE, :RE] - lowers[:IM, :IM]
    zestimates[:IM,:L] = lowers[:IM, :RE] + lowers[:RE, :IM]
    zestimates[:IM,:U] = uppers[:IM, :RE] + uppers[:RE, :IM]
    violation = 0.0
    if isnothing(zrcosts)
        violation += max(zestimates[:RE,:L] - zval[:RE], 0) + max(zval[:RE] - zestimates[:RE,:U], 0)
        violation += max(zestimates[:IM,:L] - zval[:IM], 0) + max(zval[:IM] - zestimates[:IM,:U], 0)
    else
        violation += (max(zestimates[:RE,:L] - zval[:RE], 0) - max(zval[:RE] - zestimates[:RE,:U], 0)) * zrcosts[:RE]
        violation += (max(zestimates[:IM,:L] - zval[:IM], 0) - max(zval[:IM] - zestimates[:IM,:U], 0)) * zrcosts[:IM]
    end
    return violation
end

# violation branching
function mixBranch3(stateseparator::StateSeparator, focusnode::Node, sol)
    ZBs = focusnode.ZBs
    param = stateseparator.param
    BST = stateseparator.problem.BST
    Zdims = stateseparator.problem.Zdims
    Zvals = sol.Zvals
    Zscores = []
    Zgapscores = []
    Zbdscores = []
    minbound = typemax(Float64)
    maxbound = 0.0
    maxgap = 0.0
    mingap = typemax(Float64)
    # create score matrices
    function createZscores(sys, leftsys, rightsys, retleft, retright)
        Zind = sys.Zind
        dim = Zdims[Zind]
        push!(Zgapscores, Dict(:RE=> zeros(dim, dim), :IM=> zeros(dim, dim)))
        push!(Zbdscores, Dict(:RE=> zeros(dim, dim), :IM=> zeros(dim, dim)))
        return nothing
    end
    traverseDPSBST(1, BST, createZscores)

    # get score matrices
    function getZscores(sys, leftsys, rightsys, retleft, retright)
        Zind = sys.Zind
        Zval = Zvals[Zind]
        dim = Zdims[Zind]
        if sys.sysid == -1
            leftind = leftsys.Zind
            rightind = rightsys.Zind
            # compute convexification gap
            dims = [Zdims[leftind], Zdims[rightind]]
            for tj in 1:dim
                for tk in 1:dim
                    js = cartIndex(dims, tj)
                    ks = cartIndex(dims, tk)
                    bounds = subFactorBounds(focusnode.fixvars, ZBs, (leftind, rightind), js, ks)
                    vars = Dict(:RE => [0., 0.], :IM => [0., 0.])
                    for (i, (j, k)) in enumerate(zip(js, ks))
                        subZind = i == 1 ? leftind : rightind
                        vars[:RE][i] = Zvals[subZind][:RE][j,k]
                        vars[:IM][i] = Zvals[subZind][:IM][j,k]
                    end
                    zvals = Dict(:RE => Zval[:RE][tj, tk], :IM => Zval[:IM][tj, tk])
                    # try change the bounds, and see how slacks change
                    for l in 1:2
                        # the var's part to change
                        for part in (:RE, :IM)
                            boundsdown = deepcopy(bounds)
                            boundsdown[part, :U][l] = vars[part][l]
                            boundsup = deepcopy(bounds)
                            boundsup[part, :L][l] = vars[part][l]
                            violationdown = branchViolation(vars, boundsdown, zvals)
                            violationup = branchViolation(vars, boundsup, zvals)
                            # NB: the `+ 1/6 * max(...)` term used to sit on its own
                            # continuation line, which Julia parsed as a separate
                            # statement and discarded. It is part of the score.
                            Zgapscores[l == 1 ? leftind : rightind][part][js[l], ks[l]] +=
                                5 / 6 * min(violationdown, violationup) +
                                1 / 6 * max(violationdown, violationup)
                        end
                    end
                end
            end
        end
        return nothing
    end
    traverseDPSBST(1, BST, getZscores)

    bestpart = :RE
    bestZind = 1
    bestdepth = 0
    bestj = 1
    bestk = 1
    bestscore = 0.0

    function bestcandidate(sys, leftsys, rightsys, retleft, retright)
        if sys.parent == -1
            return
        end
        Zind = sys.Zind
        dim = Zdims[Zind]
        for j in 1:dim
            for k in 1:dim
                for part in (:RE,:IM)
                    score = Zgapscores[Zind][part][j,k]
                    if score > bestscore && !haskey( focusnode.fixvars, (Zind, part, j, k))
                        bestpart = part
                        bestZind = Zind
                        bestj = j
                        bestk = k
                        bestscore = score
                        bestdepth = sys.depth
                    end
                end
            end
        end
    end
    traverseDPSBST(1, BST, bestcandidate)

    bestval = Zvals[bestZind][bestpart][bestj, bestk]
    bestval = abs(bestval) < param.feas_tol ? 0.0 : bestval
    width = ZBs[bestZind][bestpart,:U][bestj,bestk] - ZBs[bestZind][bestpart,:L][bestj,bestk]
    bddist = min(ZBs[bestZind][bestpart,:U][bestj,bestk] - bestval, bestval - ZBs[bestZind][bestpart,:L][bestj,bestk])
    bpoint = (bestpart, bestZind, bestj, bestk, bestval, ZBs[bestZind][bestpart,:L][bestj,bestk], ZBs[bestZind][bestpart,:U][bestj,bestk], width, bddist, bestdepth)
    return bpoint, bestscore
end

# violation branching + reduced cost branching
function executeBranchRules(stateseparator::StateSeparator, focusnode::Node, sol)
    bpoint, bestscore = mixBranch3(stateseparator, focusnode, sol)
    return bpoint, bestscore
end

# create subnodes given a branch node
function stateseparatorCreateBranchNodes!(stateseparator::StateSeparator, node::Node, part::Symbol, Zind::Int, i::Int, j::Int, brpoint::Float64)
    endnodeid = stateseparatorGetNNodes(stateseparator)
    delete!(stateseparator.leaves, node.nodeid)
    nodedown = Node(endnodeid + 1, node.nodeid, endnodeid + 2, node.depth + 1, true, deepcopy(node.ZBs), deepcopy(node.fixvars))
    nodeup = Node(endnodeid + 2, node.nodeid, endnodeid + 1, node.depth + 1, true, deepcopy(node.ZBs), deepcopy(node.fixvars))
    if abs( node.ZBs[Zind][(part, :U)][i, j] - node.ZBs[Zind][(part, :L)][i, j] ) < stateseparator.param.feas_tol
        brpoint = (node.ZBs[Zind][(part, :U)][i, j] + node.ZBs[Zind][(part, :L)][i, j]) / 2
    elseif abs( brpoint - nodedown.ZBs[Zind][(part, :L)][i, j] ) < stateseparator.param.feas_tol
        nodedown.fixvars[(Zind, part, i, j)] = brpoint
    elseif abs( brpoint - nodeup.ZBs[Zind][(part, :U)][i, j] ) < stateseparator.param.feas_tol
        nodeup.fixvars[(Zind, part, i, j)] = brpoint
    end
    nodedown.ZBs[Zind][(part, :U)][i, j] = brpoint
    nodeup.ZBs[Zind][(part, :L)][i, j] = brpoint
    stateseparatorAddNode!(stateseparator, nodedown)
    stateseparatorAddNode!(stateseparator, nodeup)
    node.childs = [endnodeid + 1, endnodeid + 2]
    node.isleave = false
end
"""
    stateseparatorBestNode(stateseparator)

Best-bound node selection: the unpruned leaf with the largest local dual bound,
or `nothing` when the tree is exhausted.
"""
function stateseparatorBestNode(stateseparator::StateSeparator)
    bestnodeid = -1
    bestbd = -Inf
    for leave in stateseparator.leaves
        node = stateseparator.nodes[leave]
        @assert node.isleave
        if !node.pruned && node.localdualbd > bestbd
            bestnodeid = leave
            bestbd = node.localdualbd
        end
    end
    return bestnodeid == -1 ? nothing : stateseparator.nodes[bestnodeid]
end

"""
    stateseparatorSelectNode(stateseparator)

Pick the next node to process: keep the current node while it is still a leaf,
otherwise fall back to best-bound selection.

A depth-first plunging strategy (best-child, then best-sibling, then best-bound,
gated on `plungedepth`) used to live here but sat behind an unconditional
`return stateseparatorBestNode(...)` and so never ran. It has been removed along
with its helpers; see git history if you want to revive it.
"""
function stateseparatorSelectNode(stateseparator::StateSeparator)
    if stateseparator.nodes[stateseparator.selectnode].isleave
        return stateseparator.nodes[stateseparator.selectnode]
    end
    return stateseparatorBestNode(stateseparator)
end
