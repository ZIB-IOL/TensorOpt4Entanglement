# ---------------------------------------------------------------------------
# Convex relaxations for the sBB linear-minimisation oracle.
#
# Every relaxation shares the same variable skeleton (`buildBaseRelaxation`):
#   Y      -- density matrix on the full space (the tensor being separated)
#   X_k    -- density matrix of subsystem k
#   Z_i    -- density matrix attached to node i of the bipartition tree,
#             with Z_root = Y and Z_leaf(k) = X_k
# The three builders below differ only in objective and coupling constraints.
# ---------------------------------------------------------------------------

"""
    densityVariable!(model, d) -> Dict(:RE => XR, :IM => XI)

Add a `d x d` complex density-matrix variable to `model`: Hermitian (real part
symmetric, imaginary part skew-symmetric), unit trace, and PSD through the real
embedding `[XR XI; -XI XR] ⪰ 0`.
"""
function densityVariable!(model::Model, d::Int)
    XR = @variable(model, [1:d, 1:d], Symmetric)
    XI = @variable(model, [1:d, 1:d] in SkewSymmetricMatrixSpace())
    @constraint(model, tr(XR) == 1.0)
    @constraint(model, [XR XI; -XI XR] in PSDCone())
    return Dict(:RE => XR, :IM => XI)
end

"""
    buildBaseRelaxation(problem) -> OptModel

Create the variable skeleton shared by all sBB relaxations. The caller adds the
objective and any coupling constraints.
"""
function buildBaseRelaxation(problem::Problem)
    dims, Zdims, BST = problem.dims, problem.Zdims, problem.BST
    model = Model()

    Y = densityVariable!(model, problem.dimH)

    Xs = Dict(:RE => [], :IM => [])
    for j in 1:problem.nsubs
        X = densityVariable!(model, dims[j])
        push!(Xs[:RE], X[:RE])
        push!(Xs[:IM], X[:IM])
    end

    Zs = Dict(:RE => [], :IM => [])
    function createAuxSysVars(sys, leftsys, rightsys, retleft, retright)
        dim = Zdims[sys.Zind]
        if sys.parent == -1                    # root of the tree: Z := Y
            @assert !isnothing(leftsys) && !isnothing(rightsys)
            @assert !isnothing(retleft) && !isnothing(retright)
            push!(Zs[:RE], Y[:RE])
            push!(Zs[:IM], Y[:IM])
        elseif sys.sysid != -1                 # leaf: Z := X_k
            push!(Zs[:RE], Xs[:RE][sys.sysid])
            push!(Zs[:IM], Xs[:IM][sys.sysid])
        else                                   # internal node: fresh variable
            @assert !isnothing(leftsys) && !isnothing(rightsys)
            @assert !isnothing(retleft) && !isnothing(retright)
            Z = densityVariable!(model, dim)
            push!(Zs[:RE], Z[:RE])
            push!(Zs[:IM], Z[:IM])
        end
        return dim
    end
    traverseDPSBST(1, BST, createAuxSysVars)

    return OptModel(model, Xs, Y, Zs)
end

"""
    addProximalCuts!(model, problem, Y)

Cut off the half-spaces behind `problem.proximal` (the previous incumbent),
when one has been recorded.
"""
function addProximalCuts!(model::Model, problem::Problem, Y)
    isnothing(problem.proximal) && return
    @constraint(model, dot(Y[:RE] - problem.proximal[:RE], problem.H[:RE] - problem.proximal[:RE]) >= 0)
    @constraint(model, dot(Y[:IM] - problem.proximal[:IM], problem.H[:IM] - problem.proximal[:IM]) >= 0)
    return
end

"""
    buildRelaxationTrivial(problem, cutoffbd)

Separation relaxation: maximise `<H, Y>` subject to `<H, Y> >= cutoffbd`.
Used at every sBB node.
"""
function buildRelaxationTrivial(problem::Problem, cutoffbd::Float64)
    optmodel = buildBaseRelaxation(problem)
    model, Y = optmodel.model, optmodel.Y

    t = @expression(model, dot(problem.H[:RE], Y[:RE]) + dot(problem.H[:IM], Y[:IM]))
    @objective(model, Max, t)
    @constraint(model, cutoffbd <= t)
    @constraint(model, problem.cutoffbound <= dot(problem.Hout[:RE], Y[:RE]) + dot(problem.Hout[:IM], Y[:IM]))
    addProximalCuts!(model, problem, Y)

    return optmodel
end

"""
    buildRelaxationThreshold(problem)

White-noise mixing threshold relaxation: minimise `t` such that
`(1 - t) * Hout + t * I/d̄` is in the relaxed separable set. This is the RLT /
DDPS+ lower bound of the paper.
"""
function buildRelaxationThreshold(problem::Problem)
    optmodel = buildBaseRelaxation(problem)
    model, Y = optmodel.model, optmodel.Y
    dimH = problem.dimH

    Min = Dict(:RE => Matrix(Diagonal(ones(dimH) / dimH)), :IM => zeros(dimH, dimH))
    @variable(model, 1 >= t >= 0)
    @constraint(model,
        ((1 - t) * problem.Hout[:RE] + t * Min[:RE]) + im * ((1 - t) * problem.Hout[:IM] + t * Min[:IM])
        .== Y[:RE] + im * Y[:IM])
    @objective(model, Min, t)

    return optmodel
end

"""
    buildRelaxationBound(problem, cutoffbd, eZind, epart, ej, ek, direction, globalobbt)

Optimisation-based bound tightening: minimise (`:L`) or maximise (`:U`) the
single entry `Z[eZind][epart][ej, ek]`. Only used when `param.max_obbt > 0`.
"""
function buildRelaxationBound(problem::Problem, cutoffbd::Float64, eZind, epart, ej, ek, direction, globalobbt)
    optmodel = buildBaseRelaxation(problem)
    model, Y, Zs = optmodel.model, optmodel.Y, optmodel.Zs

    if !globalobbt
        @constraint(model, cutoffbd <= dot(problem.H[:RE], Y[:RE]) + dot(problem.H[:IM], Y[:IM]))
        @constraint(model, problem.cutoffbound <= dot(problem.Hout[:RE], Y[:RE]) + dot(problem.Hout[:IM], Y[:IM]))
    end
    @objective(model, Min, direction == :L ? Zs[epart][eZind][ej, ek] : -Zs[epart][eZind][ej, ek])
    addProximalCuts!(model, problem, Y)

    return optmodel
end

function applyBounds(stateseparator::StateSeparator, optmodel::OptModel, focusnode::Node)
    model = optmodel.model
    ZBs = focusnode.ZBs
    Zdims = stateseparator.problem.Zdims
    Zs = optmodel.Zs

    function applyBoundSys(sys, leftsys, rightsys, retleft, retright)
        Zind = sys.Zind
        ZB = ZBs[Zind]
        dim = Zdims[Zind]
        # compute convexification gap
        for tj in 1:dim
            for tk in 1:dim
                if haskey( focusnode.fixvars, (Zind, :RE, tj, tk) )
                    @constraints(model, begin
                        Zs[:RE][Zind][tj, tk] ==  focusnode.fixvars[(Zind, :RE, tj, tk)]
                    end)
                else
                    @constraints(model, begin
                        Zs[:RE][Zind][tj, tk] >= ZB[(:RE,:L)][tj, tk]
                        Zs[:RE][Zind][tj, tk] <= ZB[(:RE,:U)][tj, tk]
                    end)
                end
                if haskey( focusnode.fixvars, (Zind, :IM, tj, tk))
                    @constraints(model, begin
                        Zs[:IM][Zind][tj, tk] ==  focusnode.fixvars[(Zind, :IM, tj, tk)]
                    end)
                else
                    @constraints(model, begin
                        Zs[:IM][Zind][tj, tk] >= ZB[(:IM,:L)][tj, tk]
                        Zs[:IM][Zind][tj, tk] <= ZB[(:IM,:U)][tj, tk]
                    end)
                end
            end
        end
        return nothing
    end

    traverseDPSBST(1, stateseparator.problem.BST, applyBoundSys)
end

function addTensorMcCormickConstraints(stateseparator::StateSeparator, optmodel::OptModel, focusnode::Node)
    model = optmodel.model
    BST = stateseparator.problem.BST
    ZRs = optmodel.Zs[:RE]
    ZIs = optmodel.Zs[:IM]
    Zdims = stateseparator.problem.Zdims

    function addAuxPSDConstraints(sys, leftsys, rightsys, retleft, retright)
        if sys.sysid == -1
            Zind = sys.Zind
            leftZind = leftsys.Zind
            rightZind = rightsys.Zind
            leftdim = Zdims[leftZind]
            rightdim = Zdims[rightZind]
            dim = Zdims[Zind]
            liftleftR = @expression(model, kron(ZRs[leftZind], Matrix{Float64}(I, rightdim, rightdim)))
            liftleftI = @expression(model, kron(ZIs[leftZind], Matrix{Float64}(I, rightdim, rightdim)))
            liftrightR = @expression(model, kron(Matrix{Float64}(I, leftdim, leftdim), ZRs[rightZind]))
            liftrightI = @expression(model, kron(Matrix{Float64}(I, leftdim, leftdim), ZIs[rightZind]))
            @constraint(model, [liftleftR liftleftI; -liftleftI liftleftR] - [ZRs[Zind] ZIs[Zind]; -ZIs[Zind] ZRs[Zind]] in PSDCone())
            @constraint(model, [liftrightR liftrightI; -liftrightI liftrightR] - [ZRs[Zind] ZIs[Zind]; -ZIs[Zind] ZRs[Zind]] in PSDCone())
            @constraint(model, [ZRs[Zind] ZIs[Zind]; -ZIs[Zind] ZRs[Zind]] - [liftleftR liftleftI; -liftleftI liftleftR] - [liftrightR liftrightI; -liftrightI liftrightR] + Matrix{Float64}(I, 2 * dim, 2 * dim)  in PSDCone())
            return nothing
        else
            return nothing
        end
    end
    traverseDPSBST(1, BST, addAuxPSDConstraints)
end

function addPartialTraceConstraints(stateseparator::StateSeparator, optmodel::OptModel, focusnode::Node)
    model = optmodel.model
    BST = stateseparator.problem.BST
    ZRs = optmodel.Zs[:RE]
    ZIs = optmodel.Zs[:IM]
    dims = stateseparator.problem.dims

    function addSysPartialTraceConstraints(sys, leftsys, rightsys, retleft, retright)
        if sys.sysid == -1
            Zind = sys.Zind
            sysids = sys.sysids
            sysdims = [dims[i] for i in sysids]
            function findSubsLeft(sys_, leftsys_, rightsys_, retleft_, retright_)
                Zind_ = sys_.Zind
                sys_ids = sys_.sysids
                ismatch = true
                for i in 1:length(sys_ids)
                    if sys_ids[i] != sysids[i]
                        ismatch = false
                        break
                    end
                end
                if ismatch
                    @constraint(model, ZRs[Zind_] + im * ZIs[Zind_] .== Ket.partial_trace(ZRs[Zind] + im * ZIs[Zind], length(sys_ids)+1:length(sysids), sysdims))
                end
            end
            traverseDPSBST(sys.left, BST, findSubsLeft)
            function findSubsRight(sys_, leftsys_, rightsys_, retleft_, retright_)
                Zind_ = sys_.Zind
                sys_ids = sys_.sysids
                ismatch = true
                for i in 1:length(sys_ids)
                    if sys_ids[end - i + 1] != sysids[end - i + 1]
                        ismatch = false
                        break
                    end
                end
                if ismatch
                    @constraint(model, ZRs[Zind_] + im * ZIs[Zind_] .== Ket.partial_trace(ZRs[Zind] + im * ZIs[Zind], 1:length(sysids) - length(sys_ids), sysdims))
                end
            end
            traverseDPSBST(sys.right, BST, findSubsRight)
            return nothing
        else
            return nothing
        end
    end
    traverseDPSBST(1, BST, addSysPartialTraceConstraints)
end

function addComplexMcCormickConstraints(stateseparator::StateSeparator, optmodel::OptModel, focusnode::Node)
    model = optmodel.model
    BST = stateseparator.problem.BST
    ZRs = optmodel.Zs[:RE]
    ZIs = optmodel.Zs[:IM]
    ZBs = focusnode.ZBs
    Zdims = stateseparator.problem.Zdims

    function addAuxLinearConstraints(sys, leftsys, rightsys, retleft, retright)
        if sys.sysid == -1
            Zind = sys.Zind
            leftind = leftsys.Zind
            rightind = rightsys.Zind
            dim = Zdims[Zind]
            # compute convexification gap
            dims = [Zdims[leftind], Zdims[rightind]]
            for tj in 1:dim
                for tk in 1:dim
                    js = cartIndex(dims, tj)
                    ks = cartIndex(dims, tk)
                    bounds = subFactorBounds(focusnode.fixvars, ZBs, (leftind, rightind), js, ks)
                    vars = Dict(:RE => Vector{JuMP.AffExpr}(undef, 2), :IM => Vector{JuMP.AffExpr}(undef, 2))
                    for (i, (j, k)) in enumerate(zip(js, ks))
                        subZind = i == 1 ? leftind : rightind
                        vars[:RE][i] = ZRs[subZind][j,k]
                        vars[:IM][i] = ZIs[subZind][j,k]
                    end
                    unders = Dict((:RE,:RE) => Vector{JuMP.AffExpr}(undef, 2), (:IM,:IM) => Vector{JuMP.AffExpr}(undef, 2),
                      (:RE,:IM) => Vector{JuMP.AffExpr}(undef, 2), (:IM, :RE) => Vector{JuMP.AffExpr}(undef, 2))
                    overs = Dict((:RE,:RE) => Vector{JuMP.AffExpr}(undef, 2), (:IM,:IM) => Vector{JuMP.AffExpr}(undef, 2),
                      (:RE,:IM) => Vector{JuMP.AffExpr}(undef, 2), (:IM, :RE) => Vector{JuMP.AffExpr}(undef, 2))
                    # compute affine udner and over estimators for each product of parts of variables
                    for var1part in (:RE,:IM)
                        for var2part in (:RE,:IM)
                            unders[var1part, var2part] = affine(vars[var1part][1], vars[var2part][2], bounds[var1part,:L][1], bounds[var2part,:L][2], bounds[var1part,:U][1], bounds[var2part,:U][2])
                            overs[var1part, var2part] = affine(vars[var1part][1], vars[var2part][2], bounds[var1part,:U][1], bounds[var2part,:L][2], bounds[var1part,:L][1], bounds[var2part,:U][2])
                        end
                    end
                    # add cuts
                    zvars = Dict(:RE=>ZRs[Zind][tj,tk], :IM=>ZIs[Zind][tj,tk])
                    for (a1,a2) in zip(unders[:RE, :RE], overs[:IM,:IM])
                        @constraint(model, zvars[:RE] >=  a1 - a2)
                    end
                    for (a1,a2) in zip(overs[:RE, :RE], unders[:IM,:IM])
                        @constraint(model, zvars[:RE] <=  a1 - a2 )
                    end
                    for (a1,a2) in zip(unders[:RE, :IM], unders[:IM,:RE])
                        @constraint(model, zvars[:IM] >=  a1 + a2)
                    end
                    for (a1,a2) in zip(overs[:RE, :IM], overs[:IM,:RE])
                        @constraint(model, zvars[:IM] <=  a1 + a2)
                    end
                end
            end
        end
    end
    traverseDPSBST(1, BST, addAuxLinearConstraints)
end

"""
    strengthenRelaxation(stateseparator, optmodel, focusnode; usepartial, usecomplexmc)

Tighten a base relaxation with the valid inequalities of the paper: node bounds,
tensor McCormick PSD cuts, scalar complex McCormick cuts, and partial-trace
consistency between a node and its children.
"""
function strengthenRelaxation(stateseparator::StateSeparator, optmodel::OptModel, focusnode::Node;
                              usepartial = true, usecomplexmc = true)
    applyBounds(stateseparator, optmodel, focusnode )
    addTensorMcCormickConstraints(stateseparator, optmodel, focusnode)
    if usecomplexmc
        addComplexMcCormickConstraints(stateseparator, optmodel, focusnode )
    end
    if usepartial
        addPartialTraceConstraints(stateseparator, optmodel, focusnode)
    end
end

function initRelaxationThreshold(stateseparator::StateSeparator, focusnode::Node)
    optmodel = buildRelaxationThreshold(stateseparator.problem)
    strengthenRelaxation(stateseparator, optmodel, focusnode)
    return optmodel
end

function initRelaxationNode(stateseparator::StateSeparator, focusnode::Node, primalbd::Float64)
    optmodel = buildRelaxationTrivial(stateseparator.problem, primalbd + stateseparator.param.obj_tol)
    strengthenRelaxation(stateseparator, optmodel, focusnode)
    return optmodel
end

function initRelaxationBound(stateseparator::StateSeparator, focusnode::Node, primalbd::Float64, eZind, epart, ej, ek, direction, globalobbt)
    optmodel = buildRelaxationBound(stateseparator.problem, primalbd + stateseparator.param.obj_tol, eZind, epart, ej, ek, direction, globalobbt)
    strengthenRelaxation(stateseparator, optmodel, focusnode)
    return optmodel
end
