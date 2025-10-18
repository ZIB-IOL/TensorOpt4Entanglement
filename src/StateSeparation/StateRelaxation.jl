function buildRelaxationTrivial(problem::Problem, cutoffbd::Float64)
    dimH = problem.dimH
    nsubs = problem.nsubs
    BST = problem.BST
    dims = problem.dims
    Zdims = problem.Zdims
    Treeids = problem.Treeids
    XRs = []
    XIs = []

    model = Model()

    YR = @variable(model, [1:dimH, 1:dimH], Symmetric)
    YI = @variable(model, [1:dimH, 1:dimH] in SkewSymmetricMatrixSpace())

    @constraint(model, tr(YR) == 1.0)
    @constraint(model, [YR YI; -YI YR] in PSDCone())

    for j in 1:nsubs
        subdim = dims[j]
        XR = @variable(model, [1:subdim, 1:subdim], Symmetric)
        push!(XRs, XR)
        XI = @variable(model, [1:subdim, 1:subdim] in SkewSymmetricMatrixSpace())
        push!(XIs, XI)
        @constraint(model, tr(XR) == 1.0)
        @constraint(model, [XR XI; -XI XR] in PSDCone())
    end

    #@variable(model, 1 >= p >= 0)
    #@constraint(model, (1 - p) * problem.H[:RE] + p * Matrix{Float64}(I, dimH, dimH) / dimH == YR)
    #@constraint(model, (1 - p) * problem.H[:IM] == YI)
    #@objective(model, Min, p)
    # HS norm
    Xs = Dict(:RE=>XRs, :IM=>XIs)
    Y =  Dict(:RE=>YR, :IM=>YI)
    Zs = Dict(:RE => [], :IM => [])
    Zvars = Dict(:RE => [], :IM => [])

    function createAuxSysVars(sys, leftsys, rightsys, retleft, retright)
        dim = Zdims[sys.Zind]
        if sys.parent == -1
            @assert !isnothing(leftsys)
            @assert !isnothing(rightsys)
            @assert !isnothing(retleft)
            @assert !isnothing(retright)
            push!(Zs[:RE], Y[:RE])
            push!(Zs[:IM], Y[:IM])
        elseif sys.sysid != -1
            push!(Zs[:RE], Xs[:RE][sys.sysid])
            push!(Zs[:IM], Xs[:IM][sys.sysid])
        else
            @assert !isnothing(leftsys)
            @assert !isnothing(rightsys)
            @assert !isnothing(retleft)
            @assert !isnothing(retright)
            XR = @variable(model, [1:dim, 1:dim], Symmetric)
            XI = @variable(model, [1:dim, 1:dim] in SkewSymmetricMatrixSpace())
            @constraint(model, tr(XR) == 1.0)
            @constraint(model, [XR XI; -XI XR] in PSDCone())
            push!(Zs[:RE], XR)
            push!(Zs[:IM], XI)
        end
        #zvarim = @variable(model, [1:dim, 1:dim])
        #zvarre = @variable(model, [1:dim, 1:dim])
        #@constraint(model, zvarim .==  Zs[:IM][end])
        #@constraint(model, zvarre .==  Zs[:RE][end])
        #push!(Zvars[:RE], zvarim)
        #push!(Zvars[:IM], zvarre)
        return dim
    end

    traverseDPSBST(1, BST, createAuxSysVars)


    t = @expression(model, dot(problem.H[:RE], YR) + dot(problem.H[:IM], YI) )
    @objective(model, Max, t)
    @constraint(model, cutoffbd <= t)
    @constraint(model, problem.cutoffbound <= dot(problem.Hout[:RE], YR) + dot(problem.Hout[:IM], YI))
    if ! isnothing(problem.proximal)
        @constraint(model, dot(YR - problem.proximal[:RE], problem.H[:RE] - problem.proximal[:RE]) >=0)
        @constraint(model, dot(YI - problem.proximal[:IM], problem.H[:IM] - problem.proximal[:IM]) >= 0)
    end

    optmodel = OptModel(model, Xs, Y, Zs, Zvars)

    return optmodel
end

function buildRelaxationThreshold(problem::Problem)
    dimH = problem.dimH
    nsubs = problem.nsubs
    BST = problem.BST
    dims = problem.dims
    Zdims = problem.Zdims
    Treeids = problem.Treeids
    XRs = []
    XIs = []

    model = Model()

    YR = @variable(model, [1:dimH, 1:dimH], Symmetric)
    YI = @variable(model, [1:dimH, 1:dimH] in SkewSymmetricMatrixSpace())

    @constraint(model, tr(YR) == 1.0)
    @constraint(model, [YR YI; -YI YR] in PSDCone())

    for j in 1:nsubs
        subdim = dims[j]
        XR = @variable(model, [1:subdim, 1:subdim], Symmetric)
        push!(XRs, XR)
        XI = @variable(model, [1:subdim, 1:subdim] in SkewSymmetricMatrixSpace())
        push!(XIs, XI)
        @constraint(model, tr(XR) == 1.0)
        @constraint(model, [XR XI; -XI XR] in PSDCone())
    end

    #@variable(model, 1 >= p >= 0)
    #@constraint(model, (1 - p) * problem.H[:RE] + p * Matrix{Float64}(I, dimH, dimH) / dimH == YR)
    #@constraint(model, (1 - p) * problem.H[:IM] == YI)
    #@objective(model, Min, p)
    # HS norm
    Xs = Dict(:RE=>XRs, :IM=>XIs)
    Y =  Dict(:RE=>YR, :IM=>YI)
    Zs = Dict(:RE => [], :IM => [])
    Zvars = Dict(:RE => [], :IM => [])

    function createAuxSysVars(sys, leftsys, rightsys, retleft, retright)
        dim = Zdims[sys.Zind]
        if sys.parent == -1
            @assert !isnothing(leftsys)
            @assert !isnothing(rightsys)
            @assert !isnothing(retleft)
            @assert !isnothing(retright)
            push!(Zs[:RE], Y[:RE])
            push!(Zs[:IM], Y[:IM])
        elseif sys.sysid != -1
            push!(Zs[:RE], Xs[:RE][sys.sysid])
            push!(Zs[:IM], Xs[:IM][sys.sysid])
        else
            @assert !isnothing(leftsys)
            @assert !isnothing(rightsys)
            @assert !isnothing(retleft)
            @assert !isnothing(retright)
            XR = @variable(model, [1:dim, 1:dim], Symmetric)
            XI = @variable(model, [1:dim, 1:dim] in SkewSymmetricMatrixSpace())
            @constraint(model, tr(XR) == 1.0)
            @constraint(model, [XR XI; -XI XR] in PSDCone())
            push!(Zs[:RE], XR)
            push!(Zs[:IM], XI)
        end
        #zvarim = @variable(model, [1:dim, 1:dim])
        #zvarre = @variable(model, [1:dim, 1:dim])
        #@constraint(model, zvarim .==  Zs[:IM][end])
        #@constraint(model, zvarre .==  Zs[:RE][end])
        #push!(Zvars[:RE], zvarim)
        #push!(Zvars[:IM], zvarre)
        return dim
    end

    traverseDPSBST(1, BST, createAuxSysVars)

    Min = Dict(:RE=> Matrix( Diagonal(ones(dimH) / dimH)), :IM=>zeros(dimH, dimH))

    @variable(model, 1>= t >= 0)
    @constraint(model, ( (1 - t) * problem.Hout[:RE] + t * Min[:RE]) + im *  ( (1 - t) * problem.Hout[:IM] + t *  Min[:IM]) .==  YR + im * YI)
    @objective(model, Min, t)

    optmodel = OptModel(model, Xs, Y, Zs, Zvars)

    return optmodel
end

function buildRelaxationBound(problem::Problem, cutoffbd::Float64, eZind, epart, ej, ek, direction, globalobbt)
    dimH = problem.dimH
    nsubs = problem.nsubs
    BST = problem.BST
    dims = problem.dims
    Zdims = problem.Zdims
    Treeids = problem.Treeids
    XRs = []
    XIs = []

    model = Model()

    YR = @variable(model, [1:dimH, 1:dimH], Symmetric)
    YI = @variable(model, [1:dimH, 1:dimH] in SkewSymmetricMatrixSpace())

    @constraint(model, tr(YR) == 1.0)
    @constraint(model, [YR YI; -YI YR] in PSDCone())

    for j in 1:nsubs
        subdim = dims[j]
        XR = @variable(model, [1:subdim, 1:subdim], Symmetric)
        push!(XRs, XR)
        XI = @variable(model, [1:subdim, 1:subdim] in SkewSymmetricMatrixSpace())
        push!(XIs, XI)
        @constraint(model, tr(XR) == 1.0)
        @constraint(model, [XR XI; -XI XR] in PSDCone())
    end

    #@variable(model, 1 >= p >= 0)
    #@constraint(model, (1 - p) * problem.H[:RE] + p * Matrix{Float64}(I, dimH, dimH) / dimH == YR)
    #@constraint(model, (1 - p) * problem.H[:IM] == YI)
    #@objective(model, Min, p)
    # HS norm

    Xs = Dict(:RE=>XRs, :IM=>XIs)
    Y =  Dict(:RE=>YR, :IM=>YI)

    Zs = Dict(:RE => [], :IM => [])
    Zvars = Dict(:RE => [], :IM => [])

    function createAuxSysVars(sys, leftsys, rightsys, retleft, retright)
        dim = Zdims[sys.Zind]
        if sys.parent == -1
            @assert !isnothing(leftsys)
            @assert !isnothing(rightsys)
            @assert !isnothing(retleft)
            @assert !isnothing(retright)
            push!(Zs[:RE], Y[:RE])
            push!(Zs[:IM], Y[:IM])
        elseif sys.sysid != -1
            push!(Zs[:RE], Xs[:RE][sys.sysid])
            push!(Zs[:IM], Xs[:IM][sys.sysid])
        else
            @assert !isnothing(leftsys)
            @assert !isnothing(rightsys)
            @assert !isnothing(retleft)
            @assert !isnothing(retright)
            dim = Zdims[sys.Zind]
            XR = @variable(model, [1:dim, 1:dim], Symmetric)
            XI = @variable(model, [1:dim, 1:dim] in SkewSymmetricMatrixSpace())
            @constraint(model, tr(XR) == 1.0)
            @constraint(model, [XR XI; -XI XR] in PSDCone())
            push!(Zs[:RE], XR)
            push!(Zs[:IM], XI)
        end
        #zvarim = @variable(model, [1:dim, 1:dim])
        #zvarre = @variable(model, [1:dim, 1:dim])
        #@constraint(model, zvarim .==  Zs[:IM][end])
        #@constraint(model, zvarre .==  Zs[:RE][end])
        #push!(Zvars[:RE], zvarim)
        #push!(Zvars[:IM], zvarre)
        return dim
    end

    traverseDPSBST(1, BST, createAuxSysVars)

    @variable(model, sepaobj)
    if !globalobbt
        @constraint(model, cutoffbd <= dot(problem.H[:RE], YR) + dot(problem.H[:IM], YI) )
        @constraint(model, problem.cutoffbound <= dot(problem.Hout[:RE], YR) + dot(problem.Hout[:IM], YI))
    end
    @objective(model, Min, direction == :L ? Zs[epart][eZind][ej,ek] : -Zs[epart][eZind][ej,ek])
    if ! isnothing(problem.proximal)
        @constraint(model, dot(YR - problem.proximal[:RE], problem.H[:RE] - problem.proximal[:RE]) >=0)
        @constraint(model, dot(YI - problem.proximal[:IM], problem.H[:IM] - problem.proximal[:IM]) >= 0)
    end


    optmodel = OptModel(model, Xs, Y, Zs, Zvars)

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
                #print(focusnode)
                #print(focusnode.fixvars)
                #print(haskey( focusnode.fixvars, (Zind, :RE, tj, tk) ))
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



function addPatialTraceConstraints(stateseparator::StateSeparator, optmodel::OptModel, focusnode::Node)
    model = optmodel.model
    BST = stateseparator.problem.BST
    ZRs = optmodel.Zs[:RE]
    ZIs = optmodel.Zs[:IM]
    dims = stateseparator.problem.dims

    function addSysPatialTraceConstraints(sys, leftsys, rightsys, retleft, retright)
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
    traverseDPSBST(1, BST, addSysPatialTraceConstraints)
end



function dps_constraints(
    model,
    ρ::AbstractMatrix,
    ρA::AbstractMatrix,
    ρB::AbstractMatrix,
    dims::AbstractVector{<:Integer},
    n::Integer;
    ppt::Bool = true,
    is_complex::Bool = true,
    isometry::AbstractMatrix = I(size(ρ, 1))
)
    dA, dB = dims
    ext_dims = [dA; repeat([dB], n)]

    # Dimension of the extension space w/ bosonic symmetries: A dim. + `n` copies of B
    d = dA * binomial(n + dB - 1, n)
    V = kron(I(dA), Ket.symmetric_projection(ComplexF64, dB, n; partial = true)) # Bosonic subspace isometry

    psd_cone = HermitianPSDCone()
    wrapper = Hermitian

    symmetric_meat = @variable(model, [1:d, 1:d] in psd_cone)
    lifted = wrapper(V * symmetric_meat * V')
    reducedρ = @expression(model, Ket.partial_trace(lifted, 3:n+1, ext_dims))
    @constraint(model, ρ == wrapper(reducedρ))
    reducedρA = @expression(model, Ket.partial_trace(lifted, 2:n+1, ext_dims))
    @constraint(model, ρA == wrapper(reducedρA))
    reducedρB = @expression(model, Ket.partial_trace(lifted, [i for i in 1:n+1 if i != 2], ext_dims))
    @constraint(model, ρB == wrapper(reducedρB))
    if ppt && n > 1
        for i ∈ 3:n+1
            @constraint(model, Ket.partial_transpose(lifted, 2:i, ext_dims) ∈ psd_cone)
        end
    end
end


function addTensorDPSConstraints(stateseparator::StateSeparator, optmodel::OptModel, focusnode::Node)
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
            dps_constraints(model, ZRs[Zind] + im * ZIs[Zind], ZRs[leftZind] + im * ZIs[leftZind], ZRs[rightZind] + im * ZIs[rightZind],
                [leftdim, rightdim], 1)
            return nothing
        else
            return nothing
        end
    end
    traverseDPSBST(1, BST, addAuxPSDConstraints)
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
                    # upper and lower bounds of variables' real and imag parts
                    bounds = Dict((:RE,:L) => zeros(2), (:RE,:U) => zeros(2), (:IM,:L) => zeros(2), (:IM,:U) => zeros(2))
                    # variables' real and imag parts
                    vars = Dict(:RE => Vector{JuMP.AffExpr}(undef, 2), :IM => Vector{JuMP.AffExpr}(undef, 2))
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


function addRank1Constraints(stateseparator::StateSeparator, optmodel::OptModel, focusnode::Node)
    model = optmodel.model
    BST = stateseparator.problem.BST
    ZRs = optmodel.Zs[:RE]
    ZIs = optmodel.Zs[:IM]
    ZBs = focusnode.ZBs
    Zdims = stateseparator.problem.Zdims

    function addSysRank1Constraints(sys, leftsys, rightsys, retleft, retright)
        if sys.sysid == -1
            Zind = sys.Zind
            dim = Zdims[Zind]
            for tj in 1:dim
                for tk in tj+1:dim
                    # upper and lower bounds of variables' real and imag parts
                    varmaps = Dict((tj,tj) => 1, (tj,tk) => 2, (tk,tj) => 3, (tk,tk) => 4)
                    bounds = Dict((:RE,:L) => zeros(4), (:RE,:U) => zeros(4), (:IM,:L) => zeros(4), (:IM,:U) => zeros(4))
                    # variables' real and imag parts
                    vars = Dict(:RE => Vector{JuMP.AffExpr}(undef, 4), :IM => Vector{JuMP.AffExpr}(undef, 4))
                    # get bounds and variables
                    for j1 in (tj,tk)
                        for j2 in (tj,tk)
                            i = varmaps[(j1,j2)]
                            bounds[:RE,:L][i] = ZBs[Zind][:RE,:L][j1,j2]
                            bounds[:RE,:U][i] = ZBs[Zind][:RE,:U][j1,j2]
                            bounds[:IM,:L][i] = ZBs[Zind][:IM,:L][j1,j2]
                            bounds[:IM,:U][i] = ZBs[Zind][:IM,:U][j1,j2]
                            vars[:RE][i] = ZRs[Zind][j1,j2]
                            vars[:IM][i] = ZIs[Zind][j1,j2]
                        end
                    end
                    gs = (((tj,tj), (tk,tk)), ((tj,tk), (tk,tj)))
                    estimatess = []
                    for t in 1:2
                        # Z[tj,tj] * Z[tk,tk] = Z[tj,tk] * Z[tk,tj]
                        g = gs[t]
                        vi1 = varmaps[g[1]]
                        vi2 = varmaps[g[2]]
                        unders = Dict((:RE,:RE) => Vector{JuMP.AffExpr}(undef, 2), (:IM,:IM) => Vector{JuMP.AffExpr}(undef, 2),
                        (:RE,:IM) => Vector{JuMP.AffExpr}(undef, 2), (:IM, :RE) => Vector{JuMP.AffExpr}(undef, 2))
                        overs = Dict((:RE,:RE) => Vector{JuMP.AffExpr}(undef, 2), (:IM,:IM) => Vector{JuMP.AffExpr}(undef, 2),
                        (:RE,:IM) => Vector{JuMP.AffExpr}(undef, 2), (:IM, :RE) => Vector{JuMP.AffExpr}(undef, 2))
                        # compute affine udner and over estimators for each product of parts of variables
                        for var1part in (:RE,:IM)
                            for var2part in (:RE,:IM)
                                unders[var1part, var2part] = affine(vars[var1part][vi1], vars[var2part][vi2], bounds[var1part,:L][vi1], bounds[var2part,:L][vi2], bounds[var1part,:U][vi1], bounds[var2part,:U][vi2])
                                overs[var1part, var2part] = affine(vars[var1part][vi1], vars[var2part][vi2], bounds[var1part,:U][vi1], bounds[var2part,:L][vi2], bounds[var1part,:L][vi1], bounds[var2part,:U][vi2])
                            end
                        end
                        estimates = Dict((:RE,:L)=>[], (:RE,:U)=>[],(:IM,:L)=>[], (:IM,:U)=>[])
                        for (under, over) in zip(unders[:RE,:RE], overs[:IM,:IM])
                            push!(estimates[:RE,:L], under - over)
                        end
                        for (over, under) in zip(overs[:RE,:RE], unders[:IM,:IM])
                            push!(estimates[:RE,:U], over - under)
                        end
                        for (under, under_) in zip(unders[:RE,:IM], unders[:IM,:RE])
                            push!(estimates[:IM,:L], under + under_)
                        end
                        for (over, over_) in zip(overs[:RE,:IM], overs[:IM,:RE])
                            push!(estimates[:IM,:U], over + over_)
                        end
                        push!(estimatess, estimates)
                    end
                    # Add
                    for part in (:RE,:IM)
                        for (under, over) in zip(estimatess[1][part,:L], estimatess[2][part,:U])
                            @constraint(model, under <= over)
                        end
                        for (over, under) in zip(estimatess[1][part,:U], estimatess[2][part,:L])
                            @constraint(model, under <= over)
                        end
                    end
                end
            end
        end
    end
    traverseDPSBST(1, BST, addSysRank1Constraints)
end


function addMcCormickPSDConstraints_(stateseparator::StateSeparator, optmodel::OptModel, focusnode::Node)
    model = optmodel.model
    dimH = stateseparator.problem.dimH
    nsubs = stateseparator.problem.nsubs
    dims = stateseparator.problem.dims

    sum_kron_prodR =  0.0
    sum_kron_prodI =  0.0
    Xs = optmodel.Xs
    YI = optmodel.Y[:IM]
    YR = optmodel.Y[:RE]

    for j in 1:nsubs
        # Construct the Kronecker product
        kron_prodR = 1.0
        kron_prodI = 1.0
        for l in 1:nsubs
            subdim = dims[j]
            if l == j
            kron_prodI = @expression(model, kron(kron_prodI, Xs[:IM][l]))
            kron_prodR = @expression(model, kron(kron_prodR, Xs[:RE][l]))
            else
            M = Matrix{Float64}(I, subdim, subdim)
            kron_prodI = @expression(model, kron(kron_prodI, M))
            kron_prodR = @expression(model, kron(kron_prodR, M))
            end
        end
        # Add the semidefinite constraint
        if j == 1
            sum_kron_prodR = @expression(model, kron_prodR)
            sum_kron_prodI = @expression(model, kron_prodI)
        else
            sum_kron_prodR = @expression(model, sum_kron_prodR + kron_prodR)
            sum_kron_prodI = @expression(model, sum_kron_prodI + kron_prodI)
        end
        @constraint(model, [kron_prodR kron_prodI; -kron_prodI kron_prodR] - [YR YI; -YI YR] in PSDCone())
        #print(cons, "\n")
    end
    @constraint(model, [YR YI; -YI YR] - [sum_kron_prodR sum_kron_prodI; -sum_kron_prodI sum_kron_prodR] + (nsubs - 1) * Matrix{Float64}(I, dimH * 2, dimH * 2) in PSDCone())
end

function addConstraintsFromParents(stateseparator::StateSeparator, optmodel::OptModel, focusnode::Node)
    nodes = stateseparator.nodes
    parentid = focusnode.parentid
    cuts = []
    while parentid != -1
        node = nodes[parentid]
        localcutpool = node.localcutpool
        addCutFromCutPool(stateseparator, optmodel, focusnode, localcutpool)
        parentid = node.parentid
    end
end

function strenghtenRelaxation(stateseparator::StateSeparator, optmodel::OptModel, focusnode::Node, usedps = false, usepartial = true, usecomplexmc = true, userank1 = false)
    applyBounds(stateseparator, optmodel, focusnode )
    addTensorMcCormickConstraints(stateseparator, optmodel, focusnode)
    if usecomplexmc
        addComplexMcCormickConstraints(stateseparator, optmodel, focusnode )
    end
    if usepartial
        addPatialTraceConstraints(stateseparator, optmodel, focusnode)
        #addRank1Constraints(stateseparator, optmodel, focusnode)
    end
end

function initRelaxationThreshold(stateseparator::StateSeparator, focusnode::Node, usedps = false)
    optmodel = buildRelaxationThreshold(stateseparator.problem)
    strenghtenRelaxation(stateseparator, optmodel, focusnode, usedps)
    return optmodel
end

function initRelaxationNode(stateseparator::StateSeparator, focusnode::Node, primalbd::Float64, usedps = false)
    optmodel = buildRelaxationTrivial(stateseparator.problem, primalbd + stateseparator.param.obj_tol)
    strenghtenRelaxation(stateseparator, optmodel, focusnode, usedps)
    return optmodel
end

function initRelaxationBound(stateseparator::StateSeparator, focusnode::Node, primalbd::Float64, eZind, epart, ej, ek, direction, globalobbt)
    optmodel = buildRelaxationBound(stateseparator.problem, primalbd + stateseparator.param.obj_tol, eZind, epart, ej, ek, direction, globalobbt)
    strenghtenRelaxation(stateseparator, optmodel, focusnode)
    return optmodel
end