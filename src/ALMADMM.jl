
using Zygote
using LinearAlgebra, Manifolds, ManifoldsBase
using RecursiveArrayTools
import ManifoldsBase: representation_size, manifold_dimension, inner, zero_vector, zero_vector!, retract_project!, parallel_transport_to!, log!, rand!, copy
import Manopt: max_stepsize, get_reason, get_solver_return



struct LiftModel <: AbstractManifold{ℝ}
    nrank1::Int64
    dims::Vector{Int64}
    cdims::Vector{Int64}
    nsubs::Int64
    sumdim::Int64

    function LiftModel(nrank1, dims, nsubs)
        cdims = deepcopy(dims)
        cdims = cumulativeAdd!(cdims)
        new(nrank1, dims, cdims, nsubs, reduce(+,dims))
    end
end



function manifold_dimension(M::LiftModel)
    return M.sumdim * M.nrank1 * 2
end

function representation_size(M::LiftModel)
    return (manifold_dimension(M), )
end

function zero_vector(M::LiftModel, p)
    return zeros(Float64, representation_size(M))
end

function zero_vector!(M::LiftModel, X, p)
    fill!(X, 0.)
end

function rand!(M::LiftModel, X; vector_at = nothing)
    fill!(X, 0.)
end


function inner(M::LiftModel, p, pX, pY)
    return dot(pX, pY)
end

function vecNormSquare(M::LiftModel, p)
    nrank1 = M.nrank1
    sumdim = M.sumdim
    dims = M.dims
    cdims = M.cdims
    nsubs = M.nsubs
    normsquare = zeros(Float64, nrank1, nsubs)
    normsquareprod = zeros(Float64, nrank1)
    for i in 1:nrank1
        prod = 1.0
        for j in 1:nsubs
            dim = dims[j]
            idx_start = (i - 1) * sumdim * 2 + (cdims[j] - dim) * 2 + 1
            idx_end = (i - 1) * sumdim * 2 + cdims[j] * 2
            x = p[idx_start:idx_end]
            normsquare[i, j] = dot(x, x) # xnorm2
            prod *= normsquare[i, j]
        end
        normsquareprod[i] = prod
    end
    return normsquare, normsquareprod
end

function vec2Var0(M::LiftModel, p, maxrank1 = M.nrank1)
    nrank1 = M.nrank1
    sumdim = M.sumdim
    dims = M.dims
    cdims = M.cdims
    nsubs = M.nsubs
    y = 0
    rank = min(nrank1, maxrank1)
    for i in 1:rank
        xblock = p[(i - 1) * sumdim * 2 + 1 : i * sumdim * 2]
        prod = 1
        for j in 1:nsubs
            dim = dims[j]
            x = xblock[ (cdims[j] - dim) * 2 + 1 : cdims[j] * 2]
            #if i == 1
            #    println("indices = ", (x[1], x[dim+1]))
            #end
            x = x[1:dim] + im * x[dim + 1: 2*dim]
            xx = x * x'
            prod = kron(prod, xx)
        end
        if i == 1
            y = prod
        else
            y += prod
        end
        #if i <= 3
        #    println("fobj[i, j] = ", prod[1, 1])
        #end

    end
    return y
end

function vec2Var(M::LiftModel, p, maxrank1 = M.nrank1)
    nrank1 = M.nrank1
    sumdim = M.sumdim
    dims = M.dims
    cdims = M.cdims
    nsubs = M.nsubs
    y = 0
    rank = min(nrank1, maxrank1)
    for i in 1:rank
        xblock = p[(i - 1) * sumdim * 2 + 1 : i * sumdim * 2]
        prod = 1
        for j in 1:nsubs
            dim = dims[j]
            x = xblock[ (cdims[j] - dim) * 2 + 1 : cdims[j] * 2]
            #if i == 1
            #    println("indices = ", (x[1], x[dim+1]))
            #end
            x = x[1:dim] + im * x[dim + 1: 2*dim]
            prod = kron(prod, x)
        end
        if i == 1
            y =  prod * prod'
        else
            y +=  prod * prod'
        end
        #if i <= 3
        #    println("fobj[i, j] = ", prod[1, 1])
        #end

    end
    return y
end


function trace(M::LiftModel, p)
    y = vec2Var(M, p)
    return real(tr(y))
end

function fasttrace(M::LiftModel, p)
    nrank1 = M.nrank1
    sumdim = M.sumdim
    dims = M.dims
    cdims = M.cdims
    nsubs = M.nsubs
    trace = 0.0
    for i in 1:nrank1
        prod = 1.0
        for j in 1:nsubs
            dim = dims[j]
            idx_start = (i - 1) * sumdim * 2 + (cdims[j] - dim) * 2 + 1
            idx_end = (i - 1) * sumdim * 2 + cdims[j] * 2
            x = p[idx_start:idx_end]
            xnorm2 = dot(x, x)
            prod *= xnorm2
        end
        trace += prod
    end
    return trace
end

function projtangent!(M, rg, p, g)
    gtr = gradient(p ->trace(M, p), p)[1]
    rg .= g - dot(gtr, g) / dot(gtr, gtr) * gtr
    return rg
end

function fastprojtangent!(M, rg, p, g)

    normsquare, normsquareprod = vecNormSquare(M, p)
    nrank1 = M.nrank1
    sumdim = M.sumdim
    dims = M.dims
    cdims = M.cdims
    nsubs = M.nsubs
    gtr = copy(g)
    for i in 1:nrank1
        for j in 1:nsubs
            dim = dims[j]
            idx_start = (i - 1) * sumdim * 2 + (cdims[j] - dim) * 2 + 1
            idx_end = (i - 1) * sumdim * 2 + cdims[j] * 2
            gtr[idx_start:idx_end] .= 2 * p[idx_start:idx_end] * normsquareprod[i] / ( normsquare[i,j] + 1e-8)
        end
    end
    rg .= g - dot(gtr, g) / dot(gtr, gtr) * gtr
    return rg
end


#function exp!(M::LiftModel, q, p, dp, t::Float64)
#    q .= p + t * dp
#    trc = trace(M, q)
#    q /= trc^(1/(2*M.nsubs))
#    return q
#end

function retract_project!(M::LiftModel, q, p, dp)
    t = 1.0
    q .= p + t * dp
    trc = fasttrace(M, q)
    q ./= trc^(1/(2*M.nsubs))
    return q
end

function log!(M::LiftModel, X, p, q)
    X .= q - p
    #projtangent!(M, X, p, X)
    fastprojtangent!(M, X, p, X)
    return X
end

function parallel_transport_to!(M::LiftModel, Y, p, X, q)
    # Y is the output vector, X is the vector to be transported
    Y .= X #projtangent!(M, Y, q, X)
    return Y
end

function max_stepsize(M::LiftModel)
    return 0.1
end

function make_objective_closures(M::LiftModel, dirs, Min, multipliers, rho, z, indexmap)
    # Helper: flatten manifold point to vector for AD
    function fviolate(M, p, maxrank1 = M.nrank1)
        y = vec2Var(M, p, maxrank1)
        #y /= (tr(y) + 1e-6)
        Aza = dirs * z + Min
        # y - dirs * z - Min
        violate = y - Aza
        return violate
    end

    function gdlinear!(M, vg, p, coefs)
        nrank1 = M.nrank1
        sumdim = M.sumdim
        dims = M.dims
        D = prod(dims)
        nsubs = M.nsubs

        fill!(vg, 0.0)  # Initialize gradient vector to zero

        indexlen = nsubs * 2
        forward_prods = Vector{Complex{Float64}}(undef, indexlen)
        backward_prods = Vector{Complex{Float64}}(undef, indexlen)
        for r in 1:nrank1
            for i in 1:D
                for j in 1:D
                    # get tensor index mapped back to products
                    indices = indexmap[i, j]
                    coef = coefs[i, j]
                    real_coef = real(coef)
                    imag_coef = imag(coef)
                    # get the block index in flattened tensor
                    idx_block_start = (r - 1) * sumdim * 2 + 1
                    idx_block_end =  r * sumdim * 2

                    # get the correponding block in flattened tensor
                    x = p[idx_block_start:idx_block_end]
                    forward_prods[1] = 1.0 + 0.0im
                    for k in 2:indexlen
                        real_idx, imag_idx, sign = indices[k-1]
                        forward_prods[k] = forward_prods[k-1] * (x[real_idx] + im * sign * x[imag_idx])
                    end

                    # backward product
                    backward_prods[end] = 1.0 + 0.0im
                    for k in (indexlen-1):-1:1
                        real_idx, imag_idx, sign = indices[k + 1]
                        backward_prods[k] = backward_prods[k+1] * (x[real_idx] + im * sign * x[imag_idx])
                    end

                    for (k, (real_idx, imag_idx, sign)) in enumerate(indices)
                        partial_prod = forward_prods[k] * backward_prods[k]
                        gxr = real_coef * real(partial_prod) +  imag_coef * imag(partial_prod)
                        gxi = sign * (- real_coef * imag(partial_prod) + imag_coef * real(partial_prod))
                        vg[idx_block_start + real_idx - 1] += gxr
                        vg[idx_block_start + imag_idx - 1] += gxi
                    end
                end
            end
        end
        return vg
    end

    function func(M, p, maxrank1 = M.nrank1)
        violate = fviolate(M, p, maxrank1)
        pen = dot(real(violate), real(violate)) + dot(imag(violate), imag(violate))
        f = -z
        L = dot(real(multipliers), real(violate)) + dot(imag(multipliers), imag(violate))
        return f, L, pen, violate
    end

    function al_closure(M, p, maxrank1 = M.nrank1)
        f, L, pen, _ = func(M, p, maxrank1)
        return  f + L + rho * pen
    end

    function grad_al_closure(M, p)
        #global gelapsed_time
        #start_time = time()
        vg = gradient(p -> al_closure(M, p), p)[1]
        #gelapsed_time += time() - start_time
        #println("fastprojtangent! elapsed time: $(gelapsed_time * 1000) ms")

        # projtangent!(M, vg, p, vg)
        fastprojtangent!(M, vg, p, vg)
        return vg
    end

    function fastgrad_al_closure(M, p)
        #global gelapsed_time
        #start_time = time()

        violate = fviolate(M, p)
        linearize = multipliers + 2 * rho * violate
        vg = zeros(Float64, length(p))  # Initialize gradient vector to zero
        gdlinear!(M, vg, p, linearize)

        #gelapsed_time += time() - start_time
        #println("fastprojtangent! elapsed time: $(gelapsed_time * 1000) ms")

        # projtangent!(M, vg, p, vg)
        fastprojtangent!(M, vg, p, vg)
        return vg
    end

    function fastgrad_l_closure(M, p)
        linearize = multipliers
        vg = zeros(Float64, length(p))  # Initialize gradient vector to zero
        gdlinear!(M, vg, p, linearize)
        fastprojtangent!(M, vg, p, vg)
        return vg
    end

    function l_closure(M, p)
        f, L, _, _ = func(M, p)
        return f + L
    end

    function grad_l_closure(M, p)
        vg = gradient(p -> l_closure(M, p), p)[1]
        # projtangent!(M, vg, p, vg)
        fastprojtangent!(M, vg, p, vg)
        return vg
    end

    return al_closure, fastgrad_al_closure, func, l_closure, fastgrad_l_closure
end


function GetXvals_ADMM(p, nrank1, sumdim, dims, cdims, nsubs)
    Xvalss = []
    Xsubs = []
    weights = []
    sumtrprod = 0
    for i in 1:nrank1
        xblock = p[(i - 1) * sumdim * 2 + 1 : i * sumdim * 2]
        prod = 1
        for j in 1:nsubs
            dim = dims[j]
            x = xblock[ (cdims[j] - dim) * 2 + 1 : cdims[j] * 2]
            x = x[1:dim] + im * x[dim + 1: 2*dim]
            xx = x * x'
            prod = kron(prod, xx)
        end
        trprod = tr(prod)
        scale = trprod^(1/(2 * nsubs))
        sumtrprod += trprod
        subs = []
        for j in 1:nsubs
            dim = dims[j]
            x = xblock[ (cdims[j] - dim) * 2 + 1 : cdims[j] * 2]
            x = (x[1:dim] + im * x[dim + 1: 2*dim]) / scale
            push!(subs, x)
        end
        push!(Xvalss, prod / trprod)
        push!(Xsubs, subs)
        push!(weights, real(trprod))
    end
    @assert( abs(sumtrprod - 1) < 1e-6)
    return Xvalss, Xsubs, weights
end


function SetXvals(substates, weights, nrank1, sumdim, dims, cdims, nsubs)
    p = []
    y = 0
    sumtrprod_ = 0
    k = 0
    nrank1_ = 0
    for i in 1:nrank1
        substate = substates[i]
        weight = weights[i]
        prod = 1
        if weight < 1e-9
            continue
        else
            scale = weight^(1/ (2 * nsubs))
        end

        lenp = length(p)
        for j in 1:nsubs
            append!(p, real(substate[j]) * scale)
            append!(p, imag(substate[j]) * scale)
            @assert( length(substate[j]) == dims[j])
            prod = kron(prod, substate[j] * substate[j]')
        end

        if sumdim * 2 != length(p) - lenp
            print("error dims", (sumdim, dims[1], nsubs, length(p) - lenp, length(substate[1])))
        end
        @assert(sumdim * 2 == length(p) - lenp)

        nrank1_ += 1
        xblock = p[(nrank1_ - 1) * sumdim * 2 + 1 : nrank1_ * sumdim * 2]
        prod_ = 1
        for j in 1:nsubs
            dim = dims[j]
            x = xblock[ (cdims[j] - dim) * 2 + 1 : cdims[j] * 2]
            x = x[1:dim] + im * x[dim + 1: 2*dim]
            @assert( abs(norm(x / scale - substate[j])) < 1e-6)
            xx = x * x'
            prod_ = kron(prod_, xx)
        end
        sumtrprod_ += tr(prod_)
        @assert( abs(norm(prod_ - prod * weight)) < 1e-6)

        if y == 0
            y = prod * weight
        else
            y += prod * weight
        end
    end
    @assert( abs(tr(y) - 1) < 1e-6)
    @assert( abs(sumtrprod_ - 1) < 1e-6)
    return p, nrank1_
end

function ALMADMMSolve(dims::Vector{Int64}, H, substates, weights, z, multipliers, param::Param, is_escaping = false, is_high_accuracy = false)
    #Test()
    obj_tol = param.heur_LADMM_obj_tol * ( is_high_accuracy ? 0.1 : 1)
    step_tol = param.heur_LADMM_step_tol * ( is_high_accuracy ? 0.1 : 1)
    gd_tol = param.heur_LADMM_gd_tol * ( is_high_accuracy ? 0.1 : 1)
    min_obj_tol = obj_tol * 0.1
    min_step_tol = step_tol * 0.1
    min_gd_tol = gd_tol * 0.1
    multi_bound = 1e2
    rho_max = 1e4
    rho = 1e0
    rhoscale = 0.3
    ηmax = 10

    dimH = reduce(*, dims)
    nsubs = length(dims)
    nrank1 = length(substates)
    # identity matrix
    Min = Dict(:RE=> Matrix( Diagonal(ones(dimH) / dimH)), :IM=>zeros(dimH, dimH))
    # direction matrix

    Mdir = Dict(:RE=>  real(H) - Min[:RE], :IM=> imag(H) - Min[:IM])

    sumdim = reduce(+, dims)
    cdims = deepcopy(dims)
    cdims = cumulativeAdd!(cdims)
    nsubs = length(dims)
    #seed = MersenneTwister(param.seed)

    # we can only deal with equal dimensions, recursive array tools will handle this at most one level

    @assert(length(weights) == nrank1)

    pX0, nrank1 = SetXvals(substates, weights, nrank1, sumdim, dims, cdims, nsubs)
    #print(f(M, pX0), " ", dot(real(Hbar), real(H)) + dot(imag(Hbar), imag(H)))
    #stateentry = DebugEntry(state; format = "state %1.11f %1.11f", io=stdout)
    debuginfo = [:Iteration, (:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), (:Stepsize, " Stepsize: %1.11f | "), "\n", :Stop]
    debuginfo2 = [:Iteration, (:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), "\n", :Stop]

    pX = pX0
    # Final manifold
    M = LiftModel(nrank1, dims, nsubs)

    Mdir_c = Mdir[:RE] .+ im .* Mdir[:IM]
    Min_c = Min[:RE] .+ im .* Min[:IM]
    multipliers_c = multipliers[:RE] .+ im .* multipliers[:IM]

    maxiter = is_escaping ? param.heur_LADMM_maxiter : param.heur_LADMM1_maxiter
    maxiter = is_high_accuracy ? 100000000 : maxiter
    prev_pen = 0.0
    indexmap = get_indexmap(dims)

    maxmanoptiter = min(param.heur_MANOPT_maxiter, dimH * dimH *2 +1)
    maxmanoptiter *= is_high_accuracy ? 2 : 1
    record = [:Iteration]
    for i in 1:maxiter  # adjust number of iterations as needed

        # update manifold
        myf, mygrad_f, func, _, grad_l_closure = make_objective_closures(M, Mdir_c, Min_c, multipliers_c, rho, z, indexmap)
        #f, L, pen, violate = func(M, pX)
        #y = vec2Var(M, pX)
        #y /= tr(y)
        #vg = mygrad_f(M, pX)
        #print("before, f, pen, lag, lagy, gf:", rho, " ", f, " ", pen, " ", realinner(multipliers_c, violate), " ", realinner(multipliers_c, y), " ", norm(vg), "\n" )
        state = quasi_Newton(M, myf, mygrad_f, pX; max_step_size = ηmax, debug=debuginfo, return_state=true,  record=[:Iteration],
            stopping_criterion=StopAfterIteration(maxmanoptiter) | StopWhenChangeLess(M, step_tol) | StopWhenCostChangeLess(obj_tol) | StopWhenGradientNormLess(gd_tol),
            retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())
        pX = get_solver_return(state)

        last_iteration = get_record(state, :Iteration)[end]
        # refine tolerance
        if (i == 1 || is_high_accuracy)  && last_iteration <= 3
            obj_tol = max( obj_tol / 2, min_obj_tol)
            step_tol = max( step_tol / 2, min_step_tol)
            gd_tol = max( gd_tol / 2, min_gd_tol)
        end
        #f, L, pen, violate = func(M, pX)
        #vg = mygrad_f(M, pX)
        #print("-2-rho, f, pen and g:", rho, " ", f, " ", pen , " ",  rho * pen, " ", norm(vg), "\n" )

        # update z
        y = vec2Var(M, pX)
        # a z^2 + b z + c
        y_in = y - Min_c
        a = 0
        b = -1
        b -= realinner(multipliers_c, Mdir_c)
        #print((b))
        c = realinner(multipliers_c, y_in)
        a += realinner(Mdir_c, Mdir_c) * rho
        c += realinner(y_in, y_in) * rho
        b -= realinner(Mdir_c, y_in) * rho * 2
        cstar, alstar = minimize_quadratic_on_unit_interval(a, b, c)
        z = cstar
        #print("tr(y) c and al and star ", tr(y), " ", c, " ", f + L + rho * pen, " ", cstar, " ", alstar, "\n")

        myf, mygrad_f, func = make_objective_closures(M, Mdir_c, Min_c, multipliers_c, rho, z, indexmap)
        f, L, pen, violate = func(M, pX)
        println("after: rho = $rho, f = $f, pen = $pen, alm = $(f + L + rho * pen), z = $z")

        # update multipliers
        multipliers_c += rho * violate
        #multipliers_c = clamp.(multipliers_c, -multi_bound, multi_bound)
        # Clamp real and imaginary parts separately
        multipliers_c = complex.(
            clamp.(real(multipliers_c), -multi_bound, multi_bound),
            clamp.(imag(multipliers_c), -multi_bound, multi_bound)
        )
        cur_pen = sqrt(pen)
        if  cur_pen < prev_pen * rhoscale
            rho = min(rho / rhoscale, rho_max)
        else
            rho = rho
        end
        prev_pen = cur_pen

        # check convergence
        feas_tol = obj_tol
        norm_vgl = norm(grad_l_closure(M, pX))
        print("cur_pen: ", cur_pen, " < ", feas_tol, ", norm_vgl: ", norm_vgl, "<", min_gd_tol, "\n")
        if cur_pen < feas_tol && norm_vgl < min_gd_tol
            break
        end

        # check time
        if is_time_limit_exceeded(param)
            println("Time limit exceeded, exiting...")
            break
        end
    end

    trc = fasttrace(M, pX)
    pX /= trc^(1/(2*M.nsubs))
    purestates_, substates_, weights_ = GetXvals_ADMM(pX, nrank1, sumdim, dims, cdims, nsubs)
    return purestates_, substates_, 1 - z, weights_
end

function dummySolve(dims::Vector{Int64}, H, substates, weights, z, multipliers, param::Param)
    #Test()
    nsubs = length(dims)
    nrank1 = length(substates)
    # identity matrix
    sumdim = reduce(+, dims)
    cdims = deepcopy(dims)
    cdims = cumulativeAdd!(cdims)
    nsubs = length(dims)
    #seed = MersenneTwister(param.seed)
    # we can only deal with equal dimensions, recursive array tools will handle this at most one level

    @assert(length(weights) == nrank1)
    pX0, nrank1  = SetXvals(substates, weights, nrank1, sumdim, dims, cdims, nsubs)
    purestates_, substates_, weights_ = GetXvals_ADMM(pX0, nrank1, sumdim, dims, cdims, nsubs)
    return purestates_, substates_, weights_
end