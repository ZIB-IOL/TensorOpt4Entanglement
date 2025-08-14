
using Zygote
using LinearAlgebra, Manifolds, ManifoldsBase
using RecursiveArrayTools
import ManifoldsBase: representation_size,  manifold_dimension, inner, zero_vector, zero_vector!, retract_project!, exp!, inverse_retract!, parallel_transport_to!, rand!, log!, rand, copy
import Manopt: max_stepsize, get_reason



struct CombLiftModel <: AbstractManifold{ℝ}
    nrank1::Int64
    dims::Vector{Int64}
    cdims::Vector{Int64}
    nsubs::Int64
    sumdim::Int64

    function CombLiftModel(nrank1, dims, nsubs)
        cdims = deepcopy(dims)
        cdims = cumulativeAdd!(cdims)
        new(nrank1, dims, cdims, nsubs, reduce(+,dims))
    end
end



function manifold_dimension(M::CombLiftModel)
    return M.sumdim * M.nrank1 * 2 + 1
end

function representation_size(M::CombLiftModel)
    return (manifold_dimension(M), )
end

function zero_vector(M::CombLiftModel, p)
    return zeros(Float64, representation_size(M))
end

function zero_vector!(M::CombLiftModel, X, p)
    fill!(X, 0.)
end

function rand!(M::CombLiftModel, X; vector_at = nothing)
    fill!(X, 0.)
end


function inner(M::CombLiftModel, p, pX, pY)
    return dot(pX, pY)
end

function inverse_retract!(M::CombLiftModel, X, p, q)
    X.= q - p
end

function log!(M::CombLiftModel, X, p, q)
    X.= q - p
end

function parallel_transport_to!(M::CombLiftModel, Y, p, X, q)
    Y .= X
end

function exp!(M::CombLiftModel, q, p, dp, t::Float64)
    # Project t so that p[end] + t * dp[end] ∈ [0, 1]
    if dp[end] != 0
        tmin = (0.0 - p[end]) / dp[end]
        tmax = (1.0 - p[end]) / dp[end]
        t = clamp(t, min(tmin, tmax), max(tmin, tmax))
    end
    q .= p + t * dp
    trc = trace(M, q[1:end - 1])
    q[1:end - 1] /= trc^(1/(2*M.nsubs))
    return q
end

function retract_project!(M::CombLiftModel, q, p, dp)
    return exp!(M::CombLiftModel, q, p, dp, 1.0)
end



function max_stepsize(M::CombLiftModel)
    return 0.1
end

function vec2Var(M::CombLiftModel, p)
    nrank1 = M.nrank1
    sumdim = M.sumdim
    dims = M.dims
    cdims = M.cdims
    nsubs = M.nsubs
    y = 0
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
        if i == 1
            y = prod
        else
            y += prod
        end
    end
    return y, p[end]
end

function trace(M::CombLiftModel, p)
    y, _ = vec2Var(M, p)
    return tr(y)
end


function make_objective_closures(M::CombLiftModel, dirs, Min, multipliers, lambda, z)
    # Helper: flatten manifold point to vector for AD
    function func(M, p)
        y, z = vec2Var(M, p)
        y /= (tr(y) + 1e-6)
        Aza = dirs * z + Min
        # y - dirs * z - Min
        violate = y - Aza
        pen = dot(real(violate), real(violate)) + dot(imag(violate), imag(violate))
        f = -z
        L = dot(real(multipliers), real(violate)) + dot(imag(multipliers), imag(violate))
        return f, L, pen, violate
    end

    function f_closure(M, p)
        f, L, pen, _ = func(M, p)
        return  f + L + lambda * pen
    end

    function grad_f_closure(M, p)
        vg = gradient(p -> f_closure(M, p), p)[1]
        return vg
    end

    return f_closure, grad_f_closure, func
end


function GetXvals_ALM(p, nrank1, sumdim, dims, cdims, nsubs)
    Xvalss = []
    Xsubs = []
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
    end
    @assert( abs(sumtrprod - 1) < 1e-6)
    return Xvalss, Xsubs
end


function SetXvals_ALM(substates, weights, nrank1, sumdim, dims, cdims, nsubs)
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

function ALMSolve(dims::Vector{Int64}, H, substates, weights, z, multipliers, param::Param, is_escaping)
    #Test()
    obj_tol = 1e-4
    step_tol = 1e-6
    gd_tol = 1e-5
    tau = 1
    lambda_min = 1e-1
    lambda_max = 1e7
    lambda = 1e0
    lambdascale = 2.0
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
    pX0, nrank1  = SetXvals_ALM(substates, weights, nrank1, sumdim, dims, cdims, nsubs)
    push!(pX0, z)
    #print(f(M, pX0), " ", dot(real(Hbar), real(H)) + dot(imag(Hbar), imag(H)))
    #stateentry = DebugEntry(state; format = "state %1.11f %1.11f", io=stdout)
    debuginfo = [:Iteration, (:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), (:Stepsize, " Stepsize: %1.11f | "), "\n", :Stop]
    debuginfo2 = [:Iteration, (:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), "\n", :Stop]

    pX = pX0
    # Final manifold
    M = CombLiftModel(nrank1, dims, nsubs)

    Mdir_c = Mdir[:RE] .+ im .* Mdir[:IM]
    Min_c = Min[:RE] .+ im .* Min[:IM]
    multipliers_c = multipliers[:RE] .+ im .* multipliers[:IM]

    maxiter = is_escaping ? param.heur_LADMM_maxiter : param.heur_LADMM1_maxiter
    prev_pen = 0.0
    for _ in 1:maxiter  # adjust number of iterations as needed
        # set start point
        iter = min(param.heur_MANOPT_maxiter, dimH * dimH *2 +1)

        # update manifold
        myf, mygrad_f, func = make_objective_closures(M, Mdir_c, Min_c, multipliers_c, lambda, z)
        f, L, pen, violate = func(M, pX)
        y, _ = vec2Var(M, pX)
        y /= tr(y)
        vg = mygrad_f(M, pX)
        print("before, f, pen, lag, lagy, gf:", lambda, " ", f, " ", pen, " ", realinner(multipliers_c, violate), " ", realinner(multipliers_c, y), " ", norm(vg), "\n" )
        pX = quasi_Newton(M, myf, mygrad_f, pX; max_step_size = ηmax, debug=debuginfo, stopping_criterion=StopAfterIteration(iter) | StopWhenChangeLess(M, obj_tol)  , retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())
        f, L, pen, violate = func(M, pX)
        vg = mygrad_f(M, pX)
        #print("-2-lambda, f, pen and g:", lambda, " ", f, " ", pen , " ",  lambda * pen, " ", norm(vg), "\n" )
        print("after: lambda, f, pen and alm z:", lambda, " ", f, " ", pen , " ",  f + L + lambda * pen, " ", z, "\n" )

        # update multipliers
        multipliers_c += lambda * violate
        if  sqrt(pen) > prev_pen * 0.8
            lambda = min(lambdascale * lambda, lambda_max)
        else
            lambda = lambda
        end
        prev_pen = sqrt(pen)
    end

    trc = trace(M, pX)
    pX[1:end - 1] /= trc^(1/(2*M.nsubs))
    purestates_, substates_ = GetXvals_ALM(pX, nrank1, sumdim, dims, cdims, nsubs)
    #abort()
    return purestates_, substates_
end


