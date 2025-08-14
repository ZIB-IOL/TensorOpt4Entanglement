mutable struct CGLPmodel
    model
    aR
    aI
    b
    obj
    valR
    valI
    valb
    initsize

    function CGLPmodel()
        new()
    end
end

function initialLPRelaxation!(nsubs, part, dir, xvals, fx, xbs, vertices, fvertices, added, param)
    model = Model()
    cglpmodel = CGLPmodel()
    setMosekParam(model, param)
    @variable(model, aR[1:nsubs])
    @variable(model, aI[1:nsubs])
    @variable(model, b)
    @variable(model, obj)

    # normalization
    l1normsize = 2 * nsubs + 1
    # normalization condition
    @constraint(model, [1; vcat(aR, aI) ] in MOI.NormOneCone(l1normsize) )

    @constraint(model, dot(xvals[:RE], aR) + dot(xvals[:IM], aI) + b == obj)
    if dir == :L
        @objective(model, Max, obj - fx)
    else
        @objective(model, Max, fx - obj)
    end

    # cutoff
    if dir == :L
        @constraint(model, obj >= fx + param.tol)
    else
        @constraint(model, obj <= fx - param.tol)
    end

    cglpmodel.initsize = min(param.multcut_initlpsize, length(vertices))
    cglpmodel.initsize = length(vertices)
    # add part of constraints
    i = 0
    for (i, vertex) in enumerate(Iterators.take(vertices, cglpmodel.initsize))
        added[i] = 1
        xR = [xbs[t][j] for (j,t) in enumerate(vertex[1 : nsubs])]
        xI = [xbs[t][j] for (j,t) in enumerate(vertex[nsubs + 1 : 2 * nsubs])]
        fvertex = prod([xr + im * xi for (xr, xi) in zip(xR, xI)])
        fvertex = part == :RE ? real(fvertex) : imag(fvertex)
        if dir == :L
            @constraint(model, dot(xR, aR) + dot(xI, aI) + b <= fvertex)
        else
            @constraint(model, dot(xR, aR) + dot(xI, aI) + b >= fvertex)
        end
    end
    cglpmodel.model = model
    cglpmodel.aR = aR
    cglpmodel.aI = aI
    cglpmodel.b = b
    return cglpmodel
end

function enforceConstraints!(cglpmodel, nsubs, part, dir, xbs, vertices, fvertices, added, param)
    # cuts
    separated = false
    for (i, vertex) in enumerate(Iterators.drop(vertices, cglpmodel.initsize))
        if added[i] == 1
            continue
        end
        xR = [xbs[t][j] for (j,t) in enumerate(vertex[1 : nsubs])]
        xI = [xbs[t][j] for (j,t) in enumerate(vertex[nsubs + 1 : 2 * nsubs])]
        if added[i] == 0
            fvertex = prod([xr + im * xi for (xr, xi) in zip(xR, xI)])
            fvertex = part == :RE ? real(fvertex) : imag(fvertex)
            fvertices[i] = fvertex
            added[i] = 2
        elseif added[i] == 2
            fvertex = fvertices[i]
        end
        fvertexeval = dot(xR, cglpmodel.valR) + dot(xI, cglpmodel.valI) + cglpmodel.valb
        if dir == :L && fvertexeval > fvertex - param.tol
            @constraint(cglpmodel.model, dot(xR, cglpmodel.aR) + dot(xI, cglpmodel.aI) + cglpmodel.b <= fvertex)
            added[i] = 1
            separated = true
        elseif dir == :U && fvertexeval < fvertex + param.tol
            @constraint(cglpmodel.model, dot(xR, cglpmodel.aR) + dot(xI, cglpmodel.aI) + cglpmodel.b >= fvertex)
            added[i] = 1
            separated = true
        end
    end
    return separated
end

function solveCGLP!(nsubs, part, dir, xvals, fx, xbs, vertices, fvertices, added, param)
    cglpmodel = initialLPRelaxation!(nsubs, part, dir, xvals, fx, xbs, vertices, fvertices, added, param)
    while true
        status = solveMSK(cglpmodel.model, param, true)[1]
        if status == RelaxOptimal || status == RelaxFeasible
            cglpmodel.valR = value.(cglpmodel.aR)
            cglpmodel.valI = value.(cglpmodel.aI)
            cglpmodel.valb = value.(cglpmodel.b)
            separated = enforceConstraints!(cglpmodel, nsubs, part, dir, xbs, vertices, fvertices, added, param)
            if !separated
                return cglpmodel.valR, cglpmodel.valI, cglpmodel.valb
            end
        else
            break
        end
    end
    return nothing, nothing, nothing
end

function separareMultilinearCuts(stateseparator::StateSeparator, optmodel::OptModel, focusnode::Node, rootZind, subZinds, subZdims, param::Param)
    rootdim = stateseparator.problem.Zdims[rootZind]
    ZBs = focusnode.ZBs
    Zvals = focusnode.sol.Zvals
    nsubs = length(subZinds)
    tj = 1
    tk = 1
    # compute convexification gap
    box = vcat([1:2 for i in 1:nsubs], [3:4 for i in 1:nsubs])
    vertices =  Iterators.product(box...)
    numvertices = length(vertices)
    added = Dict(:RE=> zeros(numvertices), :IM=> zeros(numvertices))
    fvertices = Dict(:RE=> zeros(numvertices), :IM=> zeros(numvertices))
    while true
        js = cartIndex(subZdims, tj)
        ks = cartIndex(subZdims, tk)
        Zijks =  [(subZinds[i], j, k) for (i, (j,k) ) in enumerate(zip(js, ks))]
        xbs = [ [ZBs[i][:RE, :L][j,k] for (i,j,k) in Zijks],
                [ZBs[i][:RE, :U][j,k] for (i,j,k) in Zijks],
                [ZBs[i][:IM, :L][j,k] for (i,j,k) in Zijks],
                [ZBs[i][:IM, :U][j,k] for (i,j,k) in Zijks]
              ]
        z = Dict(:RE => Zvals[rootZind][:RE][tj, tk], :IM => Zvals[rootZind][:IM][tj, tk])
        xvals = Dict(
            :RE => [Zvals[i][:RE][j,k]  for (i,j,k) in Zijks],
            :IM => [Zvals[i][:IM][j,k]  for (i,j,k) in Zijks]
        )
        # separate
        for part in (:RE,:IM)
            fx = z[part]
            for dir in (:L,:U)
                fill!(added[part], 0)
                valR, valI, valb = solveCGLP!(nsubs, part, dir, xvals, fx, xbs, vertices, fvertices[part], added[part], param)
                if !isnothing(valR)
                    index = Zijks
                    fid = (rootZind, tj, tk)
                    cut = Cut(valR, valI, valb, index, part, dir, fid)
                    push!(focusnode.cuts, cut)
                end
            end
        end

        if tj == rootdim && tk == rootdim
            break
        end
        tk += 1
        if tk > rootdim
            tk = 1
            tj += 1
        end
    end
end

function separateCutLocal(stateseparator, optmodel, focusnode, param)
    BST = stateseparator.problem.BST
    Zdims = stateseparator.problem.Zdims

    subZinds = []

    rootZind = -1
    found = false
    function collectZinds(sys, leftsys, rightsys, retleft, retright)
        Zind = sys.Zind
        # intilization
        if isempty(subZinds)
            push!(subZinds, Zind)
            rootZind = Zind
        end

        if isnothing(leftsys)
            return
        end
        leftind = leftsys.Zind
        rightind = rightsys.Zind

        # expand
        if !found
            replace_element!(subZinds, Zind, leftind, rightind)
        end

        if !found && length(subZinds) >= param.multcut_varsize
            found = true
        end
    end

    traverseBFSBST(1, BST, collectZinds)

    subZdims = [Zdims[Zind] for Zind in subZinds]
    separareMultilinearCuts(stateseparator, optmodel, focusnode, rootZind, subZinds, subZdims, param)
end

function applyCutsNode(optmodel, focusnode, param, enforce = false)
    Zs = optmodel.Zs
    print(length(focusnode.cuts))
    for cut in focusnode.cuts
        Zind, tj, tk = cut.fid
        xR = [Zs[:RE][i][j,k]  for (i, j, k) in cut.index]
        xI = [Zs[:IM][i][j,k]  for (i, j, k) in cut.index]
        if cut.dir == :L
            @constraint(optmodel.model, dot(xR, cut.valR) + dot(xI, cut.valI) + cut.valb <= Zs[cut.part][Zind][tj, tk] + param.tol)
        elseif cut.dir == :U
            @constraint(optmodel.model, dot(xR, cut.valR) + dot(xI, cut.valI) + cut.valb >= Zs[cut.part][Zind][tj, tk] - param.tol)
        end
    end
end

function addCuts(stateseparator, optmodel, focusnode, param)
    curnodeid = focusnode.nodeid
    while true
        curnode = stateseparator.nodes[curnodeid]
        applyCutsNode(optmodel, curnode, param, curnode.nodeid == focusnode.nodeid)
        curnodeid = curnode.parentid
        if curnodeid == -1
            break
        end
    end
end