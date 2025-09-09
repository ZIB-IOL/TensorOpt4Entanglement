
using Zygote
using LinearAlgebra, Manifolds, ManifoldsBase
using RecursiveArrayTools
import ManifoldsBase: representation_size, manifold_dimension, inner, zero_vector, zero_vector!, default_retraction_method, parallel_transport_to!, log!, rand!, copy
import Manopt: max_stepsize, get_reason, get_solver_return



struct DualLiftModel <: AbstractManifold{ℝ}
    nrank1::Int64
    dirs
    dims::Vector{Int64}
    cdims::Vector{Int64}
    nsubs::Int64
    sumdim::Int64

    function DualLiftModel(nrank1, dirs, dims, nsubs)
        cdims = deepcopy(dims)
        cdims = cumulativeAdd!(cdims)
        new(nrank1, dirs, dims, cdims, nsubs, reduce(+,dims))
    end
end



function manifold_dimension(M::DualLiftModel)
    return M.sumdim * M.nrank1 * 2
end

function representation_size(M::DualLiftModel)
    return (manifold_dimension(M), )
end

function zero_vector(M::DualLiftModel, p)
    return zeros(Float64, representation_size(M))
end

function zero_vector!(M::DualLiftModel, X, p)
    fill!(X, 0.)
end

function rand!(M::DualLiftModel, X; vector_at = nothing)
    fill!(X, 0.)
end


function inner(M::DualLiftModel, p, pX, pY)
    return dot(pX, pY)
end

function vec2Var(M::DualLiftModel, p, maxrank1 = M.nrank1)
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


function log!(M::DualLiftModel, X, p, q)
    X .= q - p
    return X
end

function parallel_transport_to!(M::DualLiftModel, Y, p, X, q)
    # Y is the output vector, X is the vector to be transported
    Y .= X #projtangent!(M, Y, q, X)
    return Y
end

function max_stepsize(M::DualLiftModel)
    return 0.1
end

function fviolate(M, p, maxrank1 = M.nrank1)
    y = vec2Var(M, p, maxrank1)
    dirs = M.dirs
    inner = dot(real(y), real(dirs)) +  dot(imag(y), imag(dirs))
    violate = inner + 1
    return y, inner, violate
end

function retract_project!(M::DualLiftModel, q, p, dp)
    t = 1.0
    q .= p + t * dp
    _, inner, violate = fviolate(M, p)
    #println("retract: inner = $inner, violate = $violate")
    if violate < 1e-7
        scale =  1 / abs(inner)^(1/(2*M.nsubs))
        #q .*= scale
    end
    return q
end

function make_objective_closures(M::DualLiftModel, dirs, Mout, indexmap)
    # Helper: flatten manifold point to vector for AD


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
        y, inner, violate = fviolate(M, p, maxrank1)
        f = norm(y -Mout)^2 / 2 # dot(real(y), real(Mout)) + dot(imag(y), imag(Mout))
        return f, inner, violate
    end

    function fastgrad_l_closure(M, p)
        y, _, violate = fviolate(M, p)
        linearize = y - Mout
        vg = zeros(Float64, length(p))  # Initialize gradient vector to zero
        gdlinear!(M, vg, p, linearize)
        return vg
    end

    function l_closure(M, p)
        f, _, _ = func(M, p)
        return f
    end

    return func, l_closure, fastgrad_l_closure, retract_project!
end


function GetXvals_DualALM(p, nrank1, sumdim, dims, cdims, nsubs)
    Xvalss = []
    Xsubs = []
    weights = []
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
        subs = []
        for j in 1:nsubs
            dim = dims[j]
            x = xblock[ (cdims[j] - dim) * 2 + 1 : cdims[j] * 2]
            x = (x[1:dim] + im * x[dim + 1: 2*dim])
            push!(subs, x)
        end
        push!(Xvalss, prod)
        push!(Xsubs, subs)
        push!(weights, 1.0)
    end
    return Xvalss, Xsubs, weights
end


mutable struct DebugDualLiftState{TIO<:IO} <: DebugAction
    io::TIO
    DebugDualLiftState(io::IO=stdout) = new{typeof(io)}(io)
end

function (d::DebugDualLiftState)(amp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    p = s.p
    M = get_manifold(amp)
    y, inner, violate = fviolate(M, p)
    (k >= 0) && (print(d.io, (inner, violate, tr(y))))
    return nothing
end


function SetXvals_DualALM(substates, weights, nrank1, sumdim, dims, cdims, nsubs)
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

function dualALMSolve(detector, dims::Vector{Int64}, H, substates, weights, multipliers, param::Param, is_escaping = false, is_high_accuracy = false)
    #Test()
    obj_tol = param.heur_LADMM_obj_tol * ( is_high_accuracy ? 0.1 : 1)
    step_tol = param.heur_LADMM_step_tol * ( is_high_accuracy ? 0.1 : 1)
    gd_tol = param.heur_LADMM_gd_tol * ( is_high_accuracy ? 0.1 : 1)
    min_obj_tol = obj_tol * 0.1
    min_step_tol = step_tol * 0.1
    min_gd_tol = gd_tol * 0.1
    multi_min = 10
    multi_bound = 1e2
    rho_max = 2e2
    rho_min = 1e-1
    rho = param.heur_LADMM_rho
    rhoscale = 0.4
    tau = 0.8
    ηmax = 10

    dimH = reduce(*, dims)
    nsubs = length(dims)
    nrank1 = length(substates)
    # identity matrix
    Min = Dict(:RE=> Matrix( Diagonal(ones(dimH) / dimH)), :IM=>zeros(dimH, dimH))
    # direction matrix
    Mdir = Dict(:RE=> real(H) - Min[:RE], :IM=> imag(H) - Min[:IM])

    sumdim = reduce(+, dims)
    cdims = deepcopy(dims)
    cdims = cumulativeAdd!(cdims)
    nsubs = length(dims)
    #seed = MersenneTwister(param.seed)

    @assert(length(weights) == nrank1)


    pX0, nrank1 = SetXvals_DualALM(substates, weights, nrank1, sumdim, dims, cdims, nsubs)
    pX00 = copy(pX0)
    #print(f(M, pX0), " ", dot(real(Hbar), real(H)) + dot(imag(Hbar), imag(H)))
    #stateentry = DebugEntry(state; format = "state %1.11f %1.11f", io=stdout)
    debuginfo = [:Iteration, (:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), (:Stepsize, " Stepsize: %1.11f | "), DebugDualLiftState(), "\n", :Stop]
    debuginfo2 = [:Iteration, (:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), "\n", :Stop]

    pX = pX0

    Mdir_c = Mdir[:RE] .+ im .* Mdir[:IM]
    Mout_c = H
    multipliers_c = multi_min

    # Final manifold
    M = DualLiftModel(nrank1, Mdir_c, dims, nsubs)

    prev_pen = 0.0
    cur_pen = 0.0
    cur_f = 0.0
    indexmap = get_indexmap(dims)

    maxmanoptiter = min( is_escaping ? param.heur_MANOPT1_maxiter : param.heur_MANOPT_maxiter, dimH * dimH *2 +1)
    maxmanoptiter *= is_high_accuracy ? 2 : 1
    record = [:Iteration]

    maxmanoptiter= 250

    # update functions
    func, l_closure, fastgrad_l_closure = make_objective_closures(M, Mdir_c, Mout_c, indexmap)
    #print("before, f, pen, lag, lagy, gf:", rho, " ", f, " ", pen, " ", realinner(multipliers_c, violate), " ", realinner(multipliers_c, y), " ", norm(vg), "\n" )
    state = quasi_Newton(M, l_closure, fastgrad_l_closure, pX; max_step_size = ηmax, debug=debuginfo, return_state=true,  record=[:Iteration],
        stopping_criterion=StopAfterIteration(maxmanoptiter) | StopWhenChangeLess(M, step_tol) ,
        retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())

    pX = get_solver_return(state)
    ypos = vec2Var(M, pX)
    inner_pos = dot(real(ypos), real(H)) + dot(imag(ypos), imag(H))

    Mout_new = Mout_c - ypos
    Mout_new = -Mout_new
    pX = pX00
    func, l_closure, fastgrad_l_closure = make_objective_closures(M, Mdir_c, Mout_new, indexmap)
    state = quasi_Newton(M, l_closure, fastgrad_l_closure, pX; max_step_size = ηmax, debug=debuginfo, return_state=true,  record=[:Iteration],
        stopping_criterion=StopAfterIteration(maxmanoptiter) | StopWhenChangeLess(M, step_tol) ,
        retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())

    pX = get_solver_return(state)
    yneg = vec2Var(M, pX)
    f, inner, violate = func(M, pX)
    inner_neg = dot(real(yneg), real(H)) + dot(imag(yneg), imag(H))
    println("after: f = $f, inner = $inner, $inner_pos, $inner_neg")

    purestates_, substates_, weights_ = GetXvals_DualALM(pX, nrank1, sumdim, dims, cdims, nsubs)
    return purestates_, substates_, cur_pen, weights_
end
