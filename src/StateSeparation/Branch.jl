# bound distance branching
function mixBranch1(stateseparator::StateSeparator, focusnode::Node, sol)
    ZBs = focusnode.ZBs
    param = stateseparator.param
    BST = stateseparator.problem.BST
    Zdims = stateseparator.problem.Zdims
    Zvals = sol.Zvals
    Zscores = []
    # create score matrices
    function  createZscores(sys, leftsys, rightsys, retleft, retright)
        Zind = sys.Zind
        dim = Zdims[Zind]
        push!(Zscores, Dict(:RE=> zeros(dim, dim), :IM=> zeros(dim, dim)))
        return nothing
    end
    traverseDPSBST(1, BST, createZscores)
    # get score matrices
    function getZscores(sys, leftsys, rightsys, retleft, retright)
        if sys.sysid == -1
            Zind = sys.Zind
            leftind = leftsys.Zind
            rightind = rightsys.Zind
            Zval = Zvals[Zind]
            dim = Zdims[Zind]
            # compute convexification gap
            dims = [Zdims[leftind], Zdims[rightind]]
            for tj in 1:dim
                for tk in 1:dim
                    js = cartIndex(dims, tj)
                    ks = cartIndex(dims, tk)
                    #print((tj, tk, js, ks))
                    wts = Dict(:RE => [0.0, 0.0], :IM => [0.0, 0.0])
                    bddists = Dict(:RE => [0.0, 0.0], :IM => [0.0, 0.0])
                    sumwts = Dict(:RE => 0.0, :IM => 0.0)
                    ybar = 1.0 + 0 * im
                    for (i, (j,k) ) in enumerate(zip(js, ks))
                        subZind = i == 1 ? leftind : rightind
                        zvalre = Zvals[subZind][:RE][j,k]
                        zvalim = Zvals[subZind][:IM][j,k]
                        zval = zvalre + im * zvalim
                        ybar *= zval
                        bddists[:RE][i] = min(ZBs[subZind][:RE,:U][j,k] - zvalre, zvalre - ZBs[subZind][:RE,:L][j,k] )
                        bddists[:IM][i] = min(ZBs[subZind][:IM,:U][j,k] - zvalim, zvalim - ZBs[subZind][:IM,:L][j,k] )
                        wts[:RE][i] = safediv( bddists[:RE][i], ZBs[subZind][:RE,:U][j,k] - ZBs[subZind][:RE,:L][j,k], param.tol)
                        wts[:IM][i] = safediv( bddists[:IM][i], ZBs[subZind][:IM,:U][j,k] - ZBs[subZind][:IM,:L][j,k], param.tol)
                        sumwts[:RE] += wts[:RE][i]
                        sumwts[:IM] += wts[:IM][i]
                    end
                    y = Zval[:RE][tj, tk] + im * Zval[:IM][tj, tk]
                    convgap = abs(y - ybar) / max(1, abs(ybar))
                    # sum up convexification gaps and discounted scores
                    for (i, (j,k) ) in enumerate(zip(js, ks))
                        subZind = i == 1 ? leftind : rightind
                        #print((convgap, convgap * safediv( wts[:RE][i], sumwts[:RE], param.tol), bddists[:RE][i],  Zscores[Zind][:RE][tj, tk] ))
                        Zscores[subZind][:RE][j, k] += convgap * safediv( wts[:RE][i], sumwts[:RE], param.tol) + 0.01 * bddists[:RE][i] + 0.1 * Zscores[Zind][:RE][tj, tk] / 4
                        Zscores[subZind][:IM][j, k] += convgap * safediv( wts[:IM][i], sumwts[:IM], param.tol) + 0.01 * bddists[:IM][i] + 0.1 * Zscores[Zind][:IM][tj, tk] / 4
                    end
                end
            end
        end
        return nothing
    end
    traverseDPSBST(1, BST, getZscores)

    bestpart = :RE
    bestZind = 1
    bestj = 1
    bestk = 1
    bestscore = 0.0

    function  bestcandidate(sys, leftsys, rightsys, retleft, retright)
        Zind = sys.Zind
        dim = Zdims[Zind]
        for j in 1:dim
            for k in 1:dim
                for part in (:RE,:IM)
                    if Zscores[Zind][part][j,k] > bestscore && !haskey( focusnode.fixvars, (Zind, part, j, k))
                        bestpart = part
                        bestZind = Zind
                        bestj = j
                        bestk = k
                        bestscore = Zscores[Zind][part][j,k]
                    end
                end
            end
        end
    end
    traverseDPSBST(1, BST, bestcandidate)
    #print(Zscores, stateseparator.problem.Treeids, stateseparator.problem.Zdims)

    bestval = Zvals[bestZind][bestpart][bestj, bestk]
    bestval = abs(bestval) < param.tol ? 0.0 : bestval
    bpoint = (bestpart, bestZind, bestj, bestk, bestval, ZBs[bestZind][bestpart,:L][bestj,bestk], ZBs[bestZind][bestpart,:U][bestj,bestk])
    #print(bpoint, (stateseparator.problem.Treeids[bestZind], BST[stateseparator.problem.Treeids[bestZind]].sysid))
    return bpoint, bestscore
end

# convexification gap branching
function mixBranch2(stateseparator::StateSeparator, focusnode::Node, sol)
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
        #push!(Zscores, Dict(:RE=> zeros(dim, dim), :IM=> zeros(dim, dim)))
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
        for j in 1:dim
            for k in 1:dim
                for part in (:RE, :IM)
                    width = ZBs[Zind][part,:U][j,k] - ZBs[Zind][part,:L][j,k]
                    if width >= 1
                        Zbdscores[Zind][part][j, k] = log10(2.1) - log10(width)
                    else
                        Zbdscores[Zind][part][j, k] = log10(2.1) + log10(max(width, 1e-3))
                    end
                    maxbound = max( maxbound, Zbdscores[Zind][part][j, k] )
                    minbound = min( minbound, Zbdscores[Zind][part][j, k] )
                end
            end
        end
        if sys.sysid == -1
            leftind = leftsys.Zind
            rightind = rightsys.Zind
            # compute convexification gap
            dims = [Zdims[leftind], Zdims[rightind]]
            for tj in 1:dim
                for tk in 1:dim
                    js = cartIndex(dims, tj)
                    ks = cartIndex(dims, tk)
                    #print((tj, tk, js, ks))
                    wts = Dict(:RE => [0.0, 0.0], :IM => [0.0, 0.0])
                    bddists = Dict(:RE => [0.0, 0.0], :IM => [0.0, 0.0])
                    sumwts = Dict(:RE => 0.0, :IM => 0.0)
                    ybar = 1.0 + 0 * im
                    for (i, (j,k) ) in enumerate(zip(js, ks))
                        subZind = i == 1 ? leftind : rightind
                        zvalre = Zvals[subZind][:RE][j,k]
                        zvalim = Zvals[subZind][:IM][j,k]
                        zval = zvalre + im * zvalim
                        ybar *= zval
                        bddists[:RE][i] = min(ZBs[subZind][:RE,:U][j,k] - zvalre, zvalre - ZBs[subZind][:RE,:L][j,k] )
                        bddists[:IM][i] = min(ZBs[subZind][:IM,:U][j,k] - zvalim, zvalim - ZBs[subZind][:IM,:L][j,k] )
                        wts[:RE][i] = safediv( bddists[:RE][i], ZBs[subZind][:RE,:U][j,k] - ZBs[subZind][:RE,:L][j,k], param.tol)
                        wts[:IM][i] = safediv( bddists[:IM][i], ZBs[subZind][:IM,:U][j,k] - ZBs[subZind][:IM,:L][j,k], param.tol)
                        sumwts[:RE] += wts[:RE][i]
                        sumwts[:IM] += wts[:IM][i]
                    end
                    y = Zval[:RE][tj, tk] + im * Zval[:IM][tj, tk]
                    convgap = abs(y - ybar) / max(1, abs(ybar))
                    # sum up convexification gaps and discounted scores
                    for (i, (j,k) ) in enumerate(zip(js, ks))
                        subZind = i == 1 ? leftind : rightind
                        #print((convgap, convgap * safediv( wts[:RE][i], sumwts[:RE], param.tol), bddists[:RE][i],  Zscores[Zind][:RE][tj, tk] ))
                        Zgapscores[subZind][:RE][j, k] += convgap * safediv( wts[:RE][i], sumwts[:RE], param.tol)
                        Zgapscores[subZind][:IM][j, k] += convgap * safediv( wts[:IM][i], sumwts[:IM], param.tol)
                    end
                end
            end
        end
        return nothing
    end
    traverseDPSBST(1, BST, getZscores)

    function getZconstants(sys, leftsys, rightsys, retleft, retright)
        Zind = sys.Zind
        dim = Zdims[Zind]
        for j in 1:dim
            for k in 1:dim
                for part in (:RE, :IM)
                    maxgap = max( maxbound, Zgapscores[Zind][part][j, k] )
                    mingap = min( minbound, Zgapscores[Zind][part][j, k] )
                end
            end
        end
    end
    traverseDPSBST(1, BST, getZconstants)

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
                    score = safediv( Zgapscores[Zind][part][j,k], maxgap - mingap, param.tol) +
                        safediv( Zbdscores[Zind][part][j,k], maxbound - minbound, param.tol)
                    #score = min(ZBs[Zind][part, :U][j,k] - Zvals[Zind][part][j,k], Zvals[Zind][part][j,k] - ZBs[Zind][part,:L][j,k] )
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
    #print(Zscores, stateseparator.problem.Treeids, stateseparator.problem.Zdims)

    bestval = Zvals[bestZind][bestpart][bestj, bestk]
    bestval = abs(bestval) < param.feas_tol ? 0.0 : bestval
    width = ZBs[bestZind][bestpart,:U][bestj,bestk] - ZBs[bestZind][bestpart,:L][bestj,bestk]
    bddist = min(ZBs[bestZind][bestpart,:U][bestj,bestk] - bestval, bestval - ZBs[bestZind][bestpart,:L][bestj,bestk])
    bpoint = (bestpart, bestZind, bestj, bestk, bestval, ZBs[bestZind][bestpart,:L][bestj,bestk], ZBs[bestZind][bestpart,:U][bestj,bestk], width, bddist, bestdepth)
    #print(bpoint, (stateseparator.problem.Treeids[bestZind], BST[stateseparator.problem.Treeids[bestZind]].sysid))
    return bpoint, bestscore
end

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
        #push!(Zscores, Dict(:RE=> zeros(dim, dim), :IM=> zeros(dim, dim)))
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
                    # upper and lower bounds of variables' real and imag parts
                    bounds = Dict((:RE,:L) => zeros(2), (:RE,:U) => zeros(2), (:IM,:L) => zeros(2), (:IM,:U) => zeros(2))
                    # variables' real and imag parts
                    vars = Dict(:RE => [0., 0.], :IM => [0., 0.])
                    # get bounds and variables
                    for (i, (j,k) ) in enumerate(zip(js, ks))
                        subZind = i == 1 ? leftind : rightind
                        if haskey( focusnode.fixvars, (subZind, :RE, j, k))
                            bounds[:RE,:L][i] = focusnode.fixvars[(subZind, :RE, j, k)]
                            bounds[:RE,:U][i] = focusnode.fixvars[(subZind, :RE, j, k)]
                        else
                            bounds[:RE,:L][i] = ZBs[subZind][:RE,:L][j,k]
                            bounds[:RE,:U][i] = ZBs[subZind][:RE,:U][j,k]
                        end
                        if haskey( focusnode.fixvars, (subZind, :IM, j, k))
                            bounds[:IM,:L][i] = focusnode.fixvars[(subZind, :IM, j, k)]
                            bounds[:IM,:U][i] = focusnode.fixvars[(subZind, :IM, j, k)]
                        else
                            bounds[:IM,:L][i] = ZBs[subZind][:IM,:L][j,k]
                            bounds[:IM,:U][i] = ZBs[subZind][:IM,:U][j,k]
                        end
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
                            Zgapscores[l == 1 ? leftind : rightind][part][js[l], ks[l]] += 5 / 6 * min(violationdown, violationup)
                            + 1 / 6 * max(violationdown, violationup)
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
                    #score = min(ZBs[Zind][part, :U][j,k] - Zvals[Zind][part][j,k], Zvals[Zind][part][j,k] - ZBs[Zind][part,:L][j,k] )
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
    #print(Zscores, stateseparator.problem.Treeids, stateseparator.problem.Zdims)

    bestval = Zvals[bestZind][bestpart][bestj, bestk]
    bestval = abs(bestval) < param.feas_tol ? 0.0 : bestval
    width = ZBs[bestZind][bestpart,:U][bestj,bestk] - ZBs[bestZind][bestpart,:L][bestj,bestk]
    bddist = min(ZBs[bestZind][bestpart,:U][bestj,bestk] - bestval, bestval - ZBs[bestZind][bestpart,:L][bestj,bestk])
    bpoint = (bestpart, bestZind, bestj, bestk, bestval, ZBs[bestZind][bestpart,:L][bestj,bestk], ZBs[bestZind][bestpart,:U][bestj,bestk], width, bddist, bestdepth)
    #print(bpoint, (stateseparator.problem.Treeids[bestZind], BST[stateseparator.problem.Treeids[bestZind]].sysid))
    return bpoint, bestscore
end


# violation branching + reduced cost branching
function mixBranch4(stateseparator::StateSeparator, focusnode::Node, sol)
    ZBs = focusnode.ZBs
    param = stateseparator.param
    BST = stateseparator.problem.BST
    Zdims = stateseparator.problem.Zdims
    Zvals = sol.Zvals
    Zrcosts = sol.Zrcosts
    Zgapscoresdown = []
    Zgapscoresup = []
    Zbdscores = []
    # create score matrices
    function createZscores(sys, leftsys, rightsys, retleft, retright)
        Zind = sys.Zind
        dim = Zdims[Zind]
        #push!(Zscores, Dict(:RE=> zeros(dim, dim), :IM=> zeros(dim, dim)))
        push!(Zgapscoresdown, Dict(:RE=> zeros(dim, dim), :IM=> zeros(dim, dim)))
        push!(Zgapscoresup, Dict(:RE=> zeros(dim, dim), :IM=> zeros(dim, dim)))
        push!(Zbdscores, Dict(:RE=> zeros(dim, dim), :IM=> zeros(dim, dim)))
        return nothing
    end
    traverseDPSBST(1, BST, createZscores)

    # get score matrices
    function getZscores(sys, leftsys, rightsys, retleft, retright)
        Zind = sys.Zind
        Zval = Zvals[Zind]
        Zrcost = Zrcosts[Zind]
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
                    # upper and lower bounds of variables' real and imag parts
                    bounds = Dict((:RE,:L) => zeros(2), (:RE,:U) => zeros(2), (:IM,:L) => zeros(2), (:IM,:U) => zeros(2))
                    # variables' real and imag parts
                    vars = Dict(:RE => [0., 0.], :IM => [0., 0.])
                    # get bounds and variables
                    for (i, (j,k) ) in enumerate(zip(js, ks))
                        subZind = i == 1 ? leftind : rightind
                        if haskey( focusnode.fixvars, (subZind, :RE, j, k))
                            bounds[:RE,:L][i] = focusnode.fixvars[(subZind, :RE, j, k)]
                            bounds[:RE,:U][i] = focusnode.fixvars[(subZind, :RE, j, k)]
                        else
                            bounds[:RE,:L][i] = ZBs[subZind][:RE,:L][j,k]
                            bounds[:RE,:U][i] = ZBs[subZind][:RE,:U][j,k]
                        end
                        if haskey( focusnode.fixvars, (subZind, :IM, j, k) )
                            bounds[:IM,:L][i] = focusnode.fixvars[(subZind, :IM, j, k)]
                            bounds[:IM,:U][i] = focusnode.fixvars[(subZind, :IM, j, k)]
                        else
                            bounds[:IM,:L][i] = ZBs[subZind][:IM,:L][j,k]
                            bounds[:IM,:U][i] = ZBs[subZind][:IM,:U][j,k]
                        end
                        vars[:RE][i] = Zvals[subZind][:RE][j,k]
                        vars[:IM][i] = Zvals[subZind][:IM][j,k]
                    end
                    zvals = Dict(:RE => Zval[:RE][tj, tk], :IM => Zval[:IM][tj, tk])
                    zrcosts = Dict(:RE => Zrcost[:RE][tj, tk], :IM => Zrcost[:IM][tj, tk])
                    # try change the bounds, and see how slacks change
                    for l in 1:2
                        # the var's part to change
                        for part in (:RE, :IM)
                            boundsdown = deepcopy(bounds)
                            boundsdown[part, :U][l] = vars[part][l]
                            boundsup = deepcopy(bounds)
                            boundsup[part, :L][l] = vars[part][l]
                            violationdown = branchViolation(vars, boundsdown, zvals, zrcosts)
                            violationup = branchViolation(vars, boundsup, zvals, zrcosts)
                            Zgapscoresdown[l == 1 ? leftind : rightind][part][js[l], ks[l]] += violationdown
                            Zgapscoresup[l == 1 ? leftind : rightind][part][js[l], ks[l]] += violationup
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
                    costleft = max(Zgapscoresdown[Zind][part][j,k], 0.0)
                    costright = max(Zgapscoresup[Zind][part][j,k], 0.0)
                    score = 5 / 6 * min(costleft, costright) + 1 / 6 * max(costleft, costright)
                    #score = min(ZBs[Zind][part, :U][j,k] - Zvals[Zind][part][j,k], Zvals[Zind][part][j,k] - ZBs[Zind][part,:L][j,k] )
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
    #print(Zscores, stateseparator.problem.Treeids, stateseparator.problem.Zdims)

    bestval = Zvals[bestZind][bestpart][bestj, bestk]
    bestval = abs(bestval) < param.feas_tol ? 0.0 : bestval
    width = ZBs[bestZind][bestpart,:U][bestj,bestk] - ZBs[bestZind][bestpart,:L][bestj,bestk]
    bddist = min(ZBs[bestZind][bestpart,:U][bestj,bestk] - bestval, bestval - ZBs[bestZind][bestpart,:L][bestj,bestk])
    bpoint = (bestpart, bestZind, bestj, bestk, bestval, ZBs[bestZind][bestpart,:L][bestj,bestk], ZBs[bestZind][bestpart,:U][bestj,bestk], width, bddist, bestdepth)
    #print(bpoint, (stateseparator.problem.Treeids[bestZind], BST[stateseparator.problem.Treeids[bestZind]].sysid))
    return bpoint, bestscore
end



function mixBranchProject(stateseparator::StateSeparator, focusnode::Node, sol)
    param = stateseparator.param
    dimH = stateseparator.problem.dimH
    nsubs = stateseparator.problem.nsubs
    Zleafids = stateseparator.problem.Zleafids
    ZBs = focusnode.ZBs
    Zdims = stateseparator.problem.Zdims
    Zvals = sol.Zvals
    Zrootid = stateseparator.problem.Zrootid

    js = fill(1, nsubs)
    ks = fill(1, nsubs)
    tj = 1
    tk = 1
    WtConvGaps = Dict(:RE => [zeros(Zdims[i], Zdims[i]) for i in Zleafids],
                      :IM => [zeros(Zdims[i], Zdims[i]) for i in Zleafids])
    WtBounds = Dict(:RE => [zeros(Zdims[i], Zdims[i]) for i in Zleafids],
                    :IM => [zeros(Zdims[i], Zdims[i]) for i in Zleafids])
    maxconvgap = 0.0
    maxbound = 0.0
    Zleafdims = [Zdims[i] for i in Zleafids]

    # compute bound scores
    for (i, (subZind, dim)) in enumerate(zip(Zleafids, Zleafdims))
        for j in 1:dim
            for k in 1:dim
                for part in (:RE, :IM)
                    width = ZBs[subZind][part,:U][j,k] - ZBs[subZind][part,:L][j,k]
                    if width >= 1
                        WtBounds[part][i][j, k] = log10(2.1) - log10(width)
                    else
                        WtBounds[part][i][j, k] = log10(2.1) + log10(max(width, 1e-3))
                    end
                    maxbound = max( maxbound, WtBounds[part][i][j, k] )
                end
            end
        end
    end

    # compute convexification gap scores
    while true
        js = cartIndex(Zleafdims, tj)
        ks = cartIndex(Zleafdims, tk)
        sumwtsRE = 0
        sumwtsIM = 0
        ybar = 1.0 + 0 * im
        wts = Dict(:RE=> fill(0.0, nsubs), :IM=> fill(0.0, nsubs))
        bddists = Dict(:RE=> fill(0.0, nsubs), :IM=> fill(0.0, nsubs))
        for (i, (j,k) ) in enumerate(zip(js, ks))
            subZind = Zleafids[i]
            zvalre = Zvals[subZind][:RE][j,k]
            zvalim = Zvals[subZind][:IM][j,k]
            zval = zvalre + im * zvalim
            ybar *= zval
            bddists[:RE][i] = min(ZBs[subZind][:RE,:U][j,k] - zvalre, zvalre - ZBs[subZind][:RE,:L][j,k] )
            bddists[:IM][i] = min(ZBs[subZind][:IM,:U][j,k] - zvalim, zvalim - ZBs[subZind][:IM,:L][j,k] )
            wts[:RE][i] = safediv( bddists[:RE][i], ZBs[subZind][:RE,:U][j,k] - ZBs[subZind][:RE,:L][j,k], param.tol)
            wts[:IM][i] = safediv( bddists[:IM][i], ZBs[subZind][:IM,:U][j,k] - ZBs[subZind][:IM,:L][j,k], param.tol)
            sumwtsRE += wts[:RE][i]
            sumwtsIM += wts[:IM][i]
        end
        y = Zvals[Zrootid][:RE][tj, tk] + im * Zvals[Zrootid][:IM][tj, tk]
        convgap = abs(y - ybar) / max(1, abs(ybar))
        for ( i, (j,k) ) in enumerate(zip(js, ks))
            WtConvGaps[:RE][i][j, k] += convgap * safediv( wts[:RE][i], sumwtsRE, param.tol)
            WtConvGaps[:IM][i][j, k] += convgap * safediv( wts[:IM][i], sumwtsIM, param.tol)
        end
        if tj == dimH && tk == dimH
            break
        end
        tk += 1
        if tk > dimH
            tk = 1
            tj += 1
        end
    end

    # normalize and mix scores
    for (i, (subZind, dim)) in enumerate(zip(Zleafids, Zleafdims))
        for j in 1:dim
            for k in 1:dim
                for part in (:RE, :IM)
                    maxconvgap = max(maxconvgap, WtConvGaps[part][i][j, k])
                end
            end
        end
    end

    for (i, (subZind, dim)) in enumerate(zip(Zleafids, Zleafdims))
        for j in 1:dim
            for k in 1:dim
                for part in (:RE, :IM)
                    WtConvGaps[part][i][j, k] = safediv(WtConvGaps[part][i][j, k], maxconvgap, param.tol)
                end
            end
        end
    end

    bestpart = :RE
    bestsys = 1
    bestj = 1
    bestk = 1
    bestscore = 0
    for i in 1:nsubs
        dim = Zdims[Zleafids[i]]
        for j in 1:dim
            for k in 1:dim
                for part in (:RE,:IM)
                    if haskey( focusnode.fixvars, (Zleafids[i], part, j, k) )
                        continue
                    end
                    if WtConvGaps[part][i][j, k] > bestscore
                        bestpart = part
                        bestsys = i
                        bestj = j
                        bestk = k
                        bestscore = WtConvGaps[part][i][j, k]
                    end
                end
            end
        end
    end

    bestZind = Zleafids[bestsys]
    bestval = Zvals[bestZind][bestpart][bestj, bestk]
    bestval = abs(bestval) < param.tol ? 0.0 : bestval
    width =  ZBs[bestZind][bestpart,:U][bestj, bestk] - ZBs[bestZind][bestpart,:L][bestj, bestk]
    bddist = min( ZBs[bestZind][bestpart,:U][bestj, bestk] - bestval, bestval - ZBs[bestZind][bestpart,:L][bestj, bestk])
    bpoint = (bestpart, bestZind, bestj, bestk, bestval, ZBs[bestZind][bestpart,:L][bestj,bestk], ZBs[bestZind][bestpart,:U][bestj,bestk], width, bddist)
    #print(bpoint, (stateseparator.problem.Treeids[bestZind], BST[stateseparator.problem.Treeids[bestZind]].sysid))
    return bpoint, bestscore, ZBs[bestZind][bestpart,:U][bestj, bestk] - ZBs[bestZind][bestpart,:L][bestj, bestk]
end

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
        #print("\n create 1", "\n")
    elseif abs( brpoint - nodedown.ZBs[Zind][(part, :L)][i, j] ) < stateseparator.param.feas_tol
        nodedown.fixvars[(Zind, part, i, j)] = brpoint
        #print("\n crearte 2", "\n")
    elseif abs( brpoint - nodeup.ZBs[Zind][(part, :U)][i, j] ) < stateseparator.param.feas_tol
        nodeup.fixvars[(Zind, part, i, j)] = brpoint
        #print("\n creat 3", "\n")
    end
    nodedown.ZBs[Zind][(part, :U)][i, j] = brpoint
    nodeup.ZBs[Zind][(part, :L)][i, j] = brpoint
    stateseparatorAddNode!(stateseparator, nodedown)
    stateseparatorAddNode!(stateseparator, nodeup)
    node.childs = [endnodeid + 1, endnodeid + 2]
    node.isleave = false
end