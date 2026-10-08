# Scalar McCormick envelopes, including collapsed (fixed) intervals.
function branchProductBounds(x, y, lx, ux, ly, uy)
    lx == ux && (x = lx)
    ly == uy && (y = ly)
    lower = max(x * ly + y * lx - lx * ly, x * uy + y * ux - ux * uy)
    upper = min(x * ly + y * ux - ux * ly, x * uy + y * lx - lx * uy)
    return lower, upper
end

function branchViolation(vars, bounds, zval, zrcosts = nothing)
    envelope(a,b) = branchProductBounds(vars[a][1], vars[b][2],
        bounds[a,:L][1], bounds[a,:U][1], bounds[b,:L][2], bounds[b,:U][2])
    lrr, urr = envelope(:RE,:RE)
    lii, uii = envelope(:IM,:IM)
    lir, uir = envelope(:IM,:RE)
    lri, uri = envelope(:RE,:IM)
    re_lower = max(lrr - uii - zval[:RE], 0.0)
    re_upper = max(zval[:RE] - (urr - lii), 0.0)
    im_lower = max(lir + lri - zval[:IM], 0.0)
    im_upper = max(zval[:IM] - (uir + uri), 0.0)
    if isnothing(zrcosts)
        return re_lower + re_upper + im_lower + im_upper
    end
    return (re_lower - re_upper) * zrcosts[:RE] +
           (im_lower - im_upper) * zrcosts[:IM]
end

# violation branching
function mixBranch3(stateseparator::StateSeparator, focusnode::Node, sol)
    ZBs = focusnode.ZBs
    param = stateseparator.param
    BST = stateseparator.problem.BST
    Zdims = stateseparator.problem.Zdims
    Zvals = sol.Zvals
    Zgapscores = []
    # create score matrices
    function createZscores(sys, leftsys, rightsys, retleft, retright)
        Zind = sys.Zind
        dim = Zdims[Zind]
        push!(Zgapscores, Dict(:RE=> zeros(dim, dim), :IM=> zeros(dim, dim)))
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
                            # These bounds are private to this matrix entry.
                            # Temporarily split one endpoint instead of copying
                            # four bound vectors for every candidate.
                            lower = bounds[part, :L][l]
                            upper = bounds[part, :U][l]
                            bounds[part, :U][l] = vars[part][l]
                            violationdown = branchViolation(vars, bounds, zvals)
                            bounds[part, :U][l] = upper
                            bounds[part, :L][l] = vars[part][l]
                            violationup = branchViolation(vars, bounds, zvals)
                            bounds[part, :L][l] = lower
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
    bestZind = 0
    bestdepth = 0
    bestj = 1
    bestk = 1
    bestscore = 0.0
    bestwidth = 0.0

    function bestcandidate(sys, leftsys, rightsys, retleft, retright)
        if sys.parent == -1
            return
        end
        Zind = sys.Zind
        dim = Zdims[Zind]
        for j in 1:dim
            for k in j:dim
                for part in (:RE,:IM)
                    score = Zgapscores[Zind][part][j,k]
                    j != k && (score += Zgapscores[Zind][part][k,j])
                    lower = ZBs[Zind][part,:L][j,k]
                    upper = ZBs[Zind][part,:U][j,k]
                    width = upper - lower
                    if width > param.feas_tol && nextfloat(lower) < upper &&
                            !haskey(focusnode.fixvars, (Zind, part, j, k)) &&
                            !haskey(focusnode.fixvars, (Zind, part, k, j)) &&
                            (score > bestscore || (score == bestscore && width > bestwidth))
                        bestpart = part
                        bestZind = Zind
                        bestj = j
                        bestk = k
                        bestscore = score
                        bestwidth = width
                        bestdepth = sys.depth
                    end
                end
            end
        end
    end
    traverseDPSBST(1, BST, bestcandidate)

    bestZind == 0 && return nothing, 0.0

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
    lower = node.ZBs[Zind][part, :L][i, j]
    upper = node.ZBs[Zind][part, :U][i, j]
    # The relative margin can round to an endpoint on tiny intervals. Require
    # a representable interior point before changing the tree, and include it
    # in the clamp so every created child strictly shrinks its parent interval.
    nextfloat(lower) < upper || return nothing
    margin = 0.1 * (upper - lower)
    brpoint = clamp(brpoint, max(lower + margin, nextfloat(lower)),
        min(upper - margin, prevfloat(upper)))
    endnodeid = stateseparatorGetNNodes(stateseparator)
    delete!(stateseparator.leaves, node.nodeid)
    nodedown = Node(endnodeid + 1, node.nodeid, endnodeid + 2, node.depth + 1, true, deepcopy(node.ZBs), deepcopy(node.fixvars))
    nodeup = Node(endnodeid + 2, node.nodeid, endnodeid + 1, node.depth + 1, true, deepcopy(node.ZBs), deepcopy(node.fixvars))
    # Unsolved children retain a valid bound when a shared deadline interrupts
    # the next node solve. Their parent's bound applies to both subdomains.
    nodedown.localdualbd = nodeup.localdualbd = node.localdualbd
    tightenEntryBounds!(nodedown,Zind,part,i,j,lower,brpoint)
    tightenEntryBounds!(nodeup,Zind,part,i,j,brpoint,upper)
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
