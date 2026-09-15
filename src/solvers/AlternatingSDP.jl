"""
    AlternateModel

One restricted SDP of the alternating method: `Xvars` are the free subsystem
density matrices (one per rank-one component), `ysvars` their lifted tensors,
`z` the interpolation variable, and `constensor` the coupling constraints whose
duals drive the pricing step.
"""
struct AlternateModel
    model::Model
    Xvars
    ysvars
    z
    constensor
end

"""
    solveAlternate(alternatemodel, param, silent = true)

Solve one restricted SDP. Returns `(relaxstatus, solver_status, sol)`; `sol`
carries the duals needed for pricing even when the model is infeasible.
"""
function solveAlternate(alternatemodel, param::Param, silent = true)
    model = alternatemodel.model
    setMosekParam(model, param)
    if param.log_level <= 2 || silent
        set_silent(model)
    end
    optimize!(model)
    status = termination_status(model)
    primalobj = objective_value(model)
    dualobj = dual_objective_value(model)
    relaxstatus = classifyStatus(objective_sense(model), status, JuMP.primal_status(model), JuMP.dual_status(model),
                                 primalobj, dualobj;
                                 obj_tol = 1e-4,
                                 infeasible_tag = RelaxInfeasibleCertificate,
                                 unknown_is_nosolution = true)
    relaxstatus == RelaxOptimal && (dualobj = primalobj)

    duals() = Dict(
        :RE => [dual.(cons) for cons in alternatemodel.constensor[:RE]],
        :IM => [dual.(cons) for cons in alternatemodel.constensor[:IM]],
    )
    if relaxstatus == RelaxOptimal || relaxstatus == RelaxFeasible
        Xval = Dict(
            :RE => [value.(xvar) for xvar in alternatemodel.Xvars[:RE]],
            :IM => [value.(xvar) for xvar in alternatemodel.Xvars[:IM]],
        )
        ysval = Dict(
            :RE => [value.(yvar) for yvar in alternatemodel.ysvars[:RE]],
            :IM => [value.(yvar) for yvar in alternatemodel.ysvars[:IM]],
        )
        return relaxstatus, status, (Xval = Xval, ysval = ysval, Xdual = duals(),
                                     dualobj = dualobj, primalobj = primalobj)
    elseif relaxstatus == RelaxInfeasibleCertificate
        return relaxstatus, status, (Xdual = duals(), dualobj = dualobj, primalobj = primalobj)
    end
    return relaxstatus, status, nothing
end

function restrict(Min, Mdir, Xvals, nsubs, dims, nrank1, freesyss)
    model = Model()
    Xvars = Dict(:RE => [], :IM => [])
    ysvars = Dict(:RE => [], :IM => [])
    constensor = Dict(:RE => [], :IM => [])

    dimH = reduce(*, dims)
    for idx in 1:nrank1
        sys = freesyss[idx]
        subdim = dims[sys]
        XR = @variable(model, [1:subdim, 1:subdim], Symmetric)
        XI = @variable(model, [1:subdim, 1:subdim] in SkewSymmetricMatrixSpace())
        @constraint(model, [XR XI; -XI XR] in PSDCone())
        push!(Xvars[:RE], XR)
        push!(Xvars[:IM], XI)
        YR = @variable(model, [1:dimH, 1:dimH], Symmetric)
        YI = @variable(model, [1:dimH, 1:dimH] in SkewSymmetricMatrixSpace())
        push!(ysvars[:RE], YR)
        push!(ysvars[:IM], YI)
    end

    for idx in 1:nrank1
        sys = freesyss[idx]
        kron_prod_left = Matrix{ComplexF64}(I, 1, 1)
        kron_prod_right = Matrix{ComplexF64}(I, 1, 1)
        for l in 1:nsubs
            if l < sys
                kron_prod_left = kron(kron_prod_left, Xvals[idx][l])
            elseif l > sys
                kron_prod_right = kron(kron_prod_right, Xvals[idx][l])
            end
        end
        kron_result = kron(kron_prod_left, Xvars[:RE][idx] + im * Xvars[:IM][idx], kron_prod_right)
        consreal = @constraint(model, ysvars[:RE][idx] == real(kron_result))
        consimag = @constraint(model, ysvars[:IM][idx] == imag(kron_result))
        push!(constensor[:RE], consreal)
        push!(constensor[:IM], consimag)
    end

    # Construct the Kronecker product
    @constraint(model, tr(sum(ysvars[:RE][idx] + im * ysvars[:IM][idx] for idx in 1:nrank1)) == 1)

    @variable(model, 0 <= z <= 1)
    @constraint(model, sum(ysvars[:RE][idx] + im * ysvars[:IM][idx] for idx in 1:nrank1) ==  (Mdir[:RE] + im * Mdir[:IM])  * z + Min[:RE] + im * Min[:IM])

    @objective(model, Max, z)

    alternatemodel = AlternateModel(model, Xvars, ysvars, z, constensor)
    return alternatemodel
end

function price(Xdual, Xvals, nrank1, nsubs, freesyss)
    freesyss_ = copy(freesyss)
    success = false
    for idx in 1:nrank1
        maxcost = -Inf
        maxsys = -1
        for sys in 1:nsubs
            if freesyss[idx] == sys
                continue
            end
            kron_prod_left = Matrix{ComplexF64}(I, 1, 1)
            kron_prod_right = Matrix{ComplexF64}(I, 1, 1)
            for l in 1:nsubs
                if l < sys
                    kron_prod_left = kron(kron_prod_left, Xvals[idx][l])
                elseif l > sys
                    kron_prod_right = kron(kron_prod_right, Xvals[idx][l])
                end
            end
            gX = partialInnerProductMap(Matrix(kron_prod_left), Matrix(kron_prod_right), Xdual[:RE][idx] + im * Xdual[:IM][idx])
            rgX = (gX + gX') / 2
            Xval = Xvals[idx][sys]
            eigenvalues, eigvecs = eigen(Xval)
            startpos = 1
            for i in 1:length(eigenvalues)
                if real(eigenvalues[i]) > 1e-6
                    startpos = i
                    break
                end
            end
            if startpos > 1
                Upos = eigvecs[:, startpos:end]
                Uzero = eigvecs[:, 1:startpos-1]
                U = [Uzero Upos]
                rgX = U' * rgX * U # this is the gradient of the objective function
            end
            subpsd = rgX[startpos:end, startpos:end]
            subeigenvalues, subeigvecs = eigen(subpsd)
            rgX[startpos:end, startpos:end] .= subeigvecs * Diagonal(max.(real(subeigenvalues),0)) * subeigvecs'
            reducedcost = norm(rgX)
            if reducedcost > maxcost
                maxcost = reducedcost
                maxsys = sys
            end
        end
        if maxcost > 1e-6
            freesyss_[idx] = maxsys
            success = true
        end
    end
    return success, freesyss_
end

function alternateSolve(dims, H, purestates, substates, weights, param, firstrun = false)
    dimH = prod(dims)
    nsubs = length(dims)

    Min = Dict(:RE=> Matrix( Diagonal(ones(dimH) / dimH)), :IM=>zeros(dimH, dimH))
    Mdir = Dict(:RE=> real(H) - Min[:RE], :IM=> imag(H) - Min[:IM])

    maxiter = firstrun ? param.heur_alternate1_iter : param.heur_alternate_iter
    maxiter =  maxiter == -1 ? 1000000 : maxiter
    fail = 0
    maxfail =  param.heur_alternate_maxfail

    Random.seed!(param.seed)
    Xvals = [ [x *x' / (norm(x)^2) for x in substate] for (i, substate) in enumerate(substates) if weights[i] > 1e-6 ]
    nrank1 = count(w -> w > 1e-6, weights)
    freesyss = [rand(1:nsubs) for _ in 1:nrank1]

    silent = true
    prevobj = 0
    ysval = Dict(:RE => [], :IM => [])
    for i in 1:maxiter
        alternatemodel = restrict(Min, Mdir, Xvals, nsubs, dims, nrank1, freesyss)
        relaxstatus, status, sol = solveAlternate(alternatemodel, param, silent)
        if sol === nothing
            break
        end
        dualprice = false
        if relaxstatus == RelaxFeasible || relaxstatus == RelaxOptimal
            if sol.primalobj > prevobj + param.obj_tol
                prevobj = sol.primalobj
                for idx in 1:nrank1
                    sys = freesyss[idx]
                    Xvals[idx][sys] = sol.Xval[:RE][idx] + im * sol.Xval[:IM][idx]
                    if abs(tr(Xvals[idx][sys])) < 1e-6
                        Xvals[idx][sys] = Matrix( Diagonal(ones(dims[sys]) / dims[sys]) )
                    else
                        Xvals[idx][sys] /= tr(Xvals[idx][sys])
                    end
                end
                Xdual = sol.Xdual
                ysval = sol.ysval
                dualprice = true
                fail = 0
            else
                Xdual = sol.Xdual
                dualprice = true
                fail += 1
            end
        elseif relaxstatus == RelaxInfeasibleCertificate
            Xdual = sol.Xdual
            dualprice = true
            fail += 1
        end

        success = false
        if dualprice
            success, freesyss_ = price(Xdual, Xvals, nrank1, nsubs, freesyss)
            if success
                freesyss = freesyss_
            end
            # update the direction matrix
        elseif fail <= maxfail
            # Reassign freesyss to new random subsystems, ensuring no repeats for each idx
            for idx in 1:nrank1
                # Choose a random subsystem for each idx, excluding the current one
                freesyss[idx] = mod1(freesyss[idx] + 1, nsubs)
            end
        else
            break
        end

        if isTimeLimitExceeded(param)
            println("Time limit exceeded, exiting...")
            break
        end

        if fail > maxfail
            break
        end

        println("Alternate Iteration $i: relaxstatus = $relaxstatus, status = $status, prevobj = $(1-prevobj), fail = $fail, success = $success")
    end

    if prevobj > 0
        substates_ = []
        purestates_ = []
        for idx in 1:nrank1
            if  isempty(ysval[:RE]) || abs(tr( ysval[:RE][idx] + im * ysval[:IM][idx] )) < 1e-6
                continue
            end
            eigenvecs = [[] for i in 1:nsubs]
            for i in 1:nsubs
                M = Xvals[idx][i]
                vals, vecs = eigen(M)
                for j in 1:length(vals)
                    if abs(vals[j]) < 1e-6
                        continue
                    end
                    push!(eigenvecs[i], vecs[:, j])
                end
            end
            for indices in Iterators.product((1:length(eigenvecs[i]) for i in 1:nsubs)...)
                subs = [eigenvecs[i][indices[i]] for i in 1:nsubs]
                pstate = [sub * sub' for sub in subs]
                state = foldl(kron, pstate)
                tracestate = tr(state)
                state /= tracestate
                push!(substates_, subs)
                push!(purestates_, state)
            end
        end
        return 1 - prevobj, purestates_, substates_
    else
        return 1 - 0, purestates, substates
    end
end