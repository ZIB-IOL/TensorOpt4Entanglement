
using Zygote
using LinearAlgebra, Manifolds, ManifoldsBase
using Manopt
import ManifoldsBase: representation_size,  manifold_dimension, inner, zero_vector, zero_vector!, retract_project!, exp!, inverse_retract!, parallel_transport_to!, rand!, log!
import Manopt: max_stepsize, get_reason


struct TROptModel <: AbstractManifold{ℂ}
    nrank1::Int64
    dims::Vector{Int64}
    cdims::Vector{Int64}
    ndim::Int64
    sumdim::Int64
    dirs
    centers
    in
    lambda::Float64

    function TROptModel(nrank1, dims, ndim, dirs, centers, in, lambda)
        cdims = deepcopy(dims)
        cdims = cumulativeAdd!(cdims)
        new(nrank1, dims, cdims, ndim, reduce(+,dims), dirs, centers, in, lambda)
    end
end


function manifold_dimension(M::TROptModel)
    return M.sumdim * M.nrank1
end

function representation_size(M::TROptModel)
    return (manifold_dimension(M), )
end

function zero_vector(M::TROptModel, p)
    return zeros(Complex{Float64}, representation_size(M))
end

function zero_vector!(M::TROptModel, X, p)
    fill!(X, 0 + 0 * im)
end

function rand!(M::TROptModel, X; vector_at = nothing)
    fill!(X, 0 + 0 * im)
end

function inverse_retract!(M::TROptModel, X, p, q)
    X.= q - p
end

function log!(M::TROptModel, X, p, q)
    X.= q - p
end

function parallel_transport_to!(M::TROptModel, Y, p, X, q)
    Y .= X
end


function inner(M::TROptModel, p, pX, pY)
    return dot(real(pX), real(pY)) + dot(imag(pX), imag(pY))
end

function max_stepsize(M::TROptModel)
    return 10
end

function retract_project!(M::TROptModel, q, p, dp)
    q .= p + dp
    trc = trace(q, M.nrank1, M.sumdim, M.dims, M.cdims, M.ndim)
    q /= trc^(1/(2*M.ndim))
    return q
end


function exp!(M::TROptModel, q, p, dp, t::Float64)
    q .= p + t * dp
    trc = trace(q, M.nrank1, M.sumdim, M.dims, M.cdims, M.ndim)
    q /= trc^(1/(2*M.ndim))
    return q
end

function yy(xs, numxs, sumdim, dims, cdims, nsubs)
    y = 0
    for i in 1:numxs
        prod = 1
        xblock = xs[(i - 1) * sumdim + 1 : i * sumdim]
        for j in 1:nsubs
            x = xblock[cdims[j] - dims[j] + 1 : cdims[j]]
            xx = x * x'
            prod = kron(prod, xx)
        end
        if i == 1
            y = prod
        else
            y += prod
        end
    end
    return y
end


function state(xs, dirs, centers, in, numxs, sumdim, dims, cdims, nsubs)
    y = yy(xs, numxs, sumdim, dims, cdims, nsubs)
    yreal = real(y)
    yimag = imag(y)
    proj = dot(yreal - real(in), real(dirs)) + dot(yimag - imag(in), imag(dirs))
    dyreal = yreal - real(centers)
    dyimag = yimag - imag(centers)
    pen =  dot(dyreal, dyreal) + dot(dyimag, dyimag)
    return proj, pen, y
end

function ff(xs, dirs, centers, in, lambda, numxs, sumdim, dims, cdims, nsubs)
    proj, pen, _ = state(xs, dirs, centers, in, numxs, sumdim, dims, cdims, nsubs)
    return pen # -proj + lambda * pen
end

function hh(xs, dirs, lambda, numxs, sumdim, dims, cdims, nsubs)
    y = 0
    for i in 1:numxs
        prod = 1
        xblock = xs[(i - 1) * sumdim + 1 : i * sumdim]
        for j in 1:nsubs
            x = xblock[cdims[j] - dims[j] + 1 : cdims[j]]
            xx = x * x'
            prod = kron(prod, xx)
        end
        if i == 1
            y = prod
        else
            y += prod
        end
    end
    yreal = real(y)
    yimag = imag(y)
    proj = dot(yreal, real(dirs)) + dot(yimag, imag(dirs))
    pen =  dot(yreal, yreal) + dot(yimag, yimag) - proj * proj
    return pen
    #pen =  dot(yreal, yreal) + dot(yimag, yimag) - proj * proj
    #return -proj + lambda * pen
end


function state(M::TROptModel, p)
    # Compute the tensor product of the matrices A_1, ..., A_n
    proj, pen,_ = state(p, M.dirs, M.centers, M.in, M.nrank1, M.sumdim, M.dims, M.cdims, M.ndim)
    return proj, pen
end


mutable struct DebugState{TIO<:IO} <: DebugAction
    io::TIO
    DebugState(io::IO=stdout) = new{typeof(io)}(io)
end

function (d::DebugState)(amp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    p = s.p
    M = get_manifold(amp)
    proj, pen = state(M, p)
    (k >= 0) && (print(d.io, (proj,pen)))
    return nothing
end

mutable struct StopBSEarly <: StoppingCriterion
    prevpen::Float64
    tol::Float64
    reltol::Float64
    fails::Int
    StopBSEarly(tol::Float64, reltol::Float64) = new(Inf, tol, reltol, 0)
end

function get_reason(sc::StopBSEarly)
    return "early stopping\n"
end

function (se::StopBSEarly)(amp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    p = s.p
    M = get_manifold(amp)
    _, pen = state(M, p)
    fail = abs(se.prevpen - pen) < se.reltol * max(se.prevpen, pen) && pen > se.tol
    se.prevpen = pen
    if fail
        se.fails += 1
    end
    return se.fails >= 3
end


mutable struct StopPolishEarly <: StoppingCriterion
    prevproj::Float64
    reltol::Float64
    fails::Int
    StopPolishEarly(reltol::Float64) = new(Inf, reltol, 0)
end

function get_reason(sc::StopPolishEarly)
    return "early stopping\n"
end

function (se::StopPolishEarly)(amp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    p = s.p
    M = get_manifold(amp)
    proj, _ = state(M, p)
    fail = abs(se.prevproj - proj) < se.reltol * max(se.prevproj, proj)
    se.prevproj = proj
    if fail
        se.fails += 1
    end
    return se.fails >= 3
end

# Objective function: Frobenius norm of the difference
function f(M::TROptModel, p)
    # Compute the tensor product of the matrices A_1, ..., A_n
    return ff(p, M.dirs, M.centers, M.in, M.lambda, M.nrank1, M.sumdim, M.dims, M.cdims, M.ndim)
end

# Gradient of the objective function
function grad_f(M::TROptModel, p)
    # Compute the gradient of f with respect to each X_i
    # Extract the vector x_i
    vg = gradient(xs -> ff(xs, M.dirs, M.centers, M.in, M.lambda, M.nrank1, M.sumdim, M.dims,  M.cdims,  M.ndim), p)
    return vg[1]
end

function trace(xs, nrank1, sumdim, dims, cdims, nsubs)
    y = yy(xs, nrank1, sumdim, dims, cdims, nsubs)
    return real(tr(y))
end

function getXvals(xs, numxs, sumdim, dims, cdims, nsubs)
    Xvalss = []
    Xsubs = []
    for i in 1:numxs
        xblock = xs[(i - 1) * sumdim + 1 : i * sumdim]
        prod = 1
        for j in 1:nsubs
            x = xblock[cdims[j] - dims[j] + 1 : cdims[j]]
            xx = x * x'
            prod = kron(prod, xx)
        end
        trprod = tr(prod)
        prod /= trprod
        push!(Xvalss, prod)
        push!(Xsubs, xblock / trprod^(1/(2*nsubs)))
    end
    return Xvalss, Xsubs
end

function getLagraingian(Mdir, Min, purestates, param)
    model = Model()
    setMosekParam(model, param, true)
    # Decision variables
    @variable(model, lambda[1:length(purestates)] >= 0)
    @constraint(model, sum(lambda) == 1)

    sumRE = sum(lambda[i] * real(purestate) for (i, purestate) in enumerate(purestates))
    sumIM = sum(lambda[i] * imag(purestate) for (i, purestate) in enumerate(purestates))

    projRE = dot(sumRE - Min[:RE], Mdir[:RE])
    projIM = dot(sumIM - Min[:IM], Mdir[:IM])
    # Objective function
    @objective(model, Max, projRE + projIM)
    dRE = ((sumRE - Min[:RE]).* Mdir[:RE]) + Min[:RE] - sumRE
    dIM = ((sumIM - Min[:IM]).* Mdir[:IM]) + Min[:IM] - sumIM
    # constraints
    @constraint(model, LRE, 0 .== dRE)
    @constraint(model, LIM, 0 .== dIM)

    @constraint(model, AL, dot(dRE, dRE) + dot(dIM, dIM) <= 1e-6)

    # Solve the model
    status, solverstatus, primalobj, _ = solveMSK(model, param, true)
    if has_duals(model)
        dualdRE = dual(dRE)
        dualdIM = dual(dIM)
        dualAL = dual(AL)
        return primalobj, dualdRE, dualdIM, dualAL
    else
        error("Dual not available")
    end
end

function TROptSolve(dims::Vector{Int64}, H, onlycenter, param::Param)
    #Test()
    sumopt_ttol = 1e-4
    sumopt_bsrel = 1e-2
    sumopt_bstol = 1e-3
    sumopt_convtol = 5e-5
    sumopt_polishtol = 1e-7
    sumopt_polishrel = 1e-3
    sumopt_outerconvtol = 5e-5
    dimH = reduce(*, dims)
    ndim = length(dims)
    nrank1 = dimH * dimH
    Min = Dict(:RE=> Matrix( Diagonal(ones(dimH) / dimH)), :IM=>zeros(dimH, dimH))
    centers = Min[:RE] + im * Min[:IM]
    Mdir = Dict(:RE=>real(H) - Min[:RE], :IM=>imag(H) - Min[:IM])
    Mdirnorm = sqrt(dot(Mdir[:RE], Mdir[:RE]) + dot(Mdir[:IM], Mdir[:IM]))
    Mdir[:RE] /= Mdirnorm
    Mdir[:IM] /= Mdirnorm
    lambda = 1.0
    sumdim = reduce(+, dims)
    cdims = deepcopy(dims)
    cdims = cumulativeAdd!(cdims)
    #seed = MersenneTwister(param.seed)
    pX0 = randn(sumdim * nrank1) + im * randn(sumdim * nrank1)
    #pX0 = randn(seed, sumdim * nrank1) + im * randn(seed, sumdim * nrank1)
    trc = trace(pX0, nrank1, sumdim, dims, cdims, ndim)
    pX0 /= trc^(1/(2*ndim))
    #print(f(M, pX0), " ", dot(real(Hbar), real(H)) + dot(imag(Hbar), imag(H)))
    #stateentry = DebugEntry(state; format = "state %1.11f %1.11f", io=stdout)
    debuginfo = [:Iteration, (:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), (:Stepsize, " Stepsize: %1.11f | "), DebugState(), "\n", :Stop]
    debuginfo2 = [:Iteration, (:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), "\n", :Stop]
    #pX = algo(M, f, grad_f, pX0; debug=[(:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), (:Stepsize, " Stepsize: %1.11f | "), "\n", :Stop],  stopping_criterion=StopAfterIteration(param.heur_MANOPT_maxiter) | StopWhenChangeLess(M, param.tol    ), retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())
    #pX = augmented_Lagrangian_method(M, f, grad_f, pX0; g = h, grad_d = grad_h,  debug=debuginfo, stopping_criterion=StopAfterIteration(param.heur_MANOPT_maxiter) | StopWhenChangeLess(M, param.tol) | StopWhenStepsizeLess(param.tol), retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())
    #monotonesearch = false
    #return  getXvals(pX0, nrank1, sumdim, dims, cdims, ndim)
    #getLagraingian(Mdir, Min, getXvals(pX0, nrank1, sumdim, dims, cdims, ndim), param)
    if onlycenter
        centers = Min[:RE] + im * Min[:IM]
        iter = min(param.heur_MANOPT_maxiter, 20)
        M = TROptModel(nrank1, dims, length(dims), Mdir[:RE] + im * Mdir[:IM], centers, Min[:RE] + im * Min[:IM], 200)
        pX0 = quasi_Newton(M, f, grad_f, pX0; debug=debuginfo, stopping_criterion=StopAfterIteration(iter) | StopWhenChangeLess(M, sumopt_convtol) | StopWhenStepsizeLess(sumopt_convtol), retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())
        return pX0
    else
        # Bisection line search for maximum point
        left = 0.0
        right = 1.0
        bestpoint =  Min[:RE] + im * Min[:IM]
        bestt = 0.0
        bestsol = nothing
        pX = pX0
        for _ in 1:20  # adjust number of iterations as needed
            if abs(right - left) < sumopt_ttol
                break
            end
            mid = (left + right) / 2
            # set start point
            pstart = pX
            if !isnothing(bestsol)
                pstart = bestsol
            end
            iter = min(param.heur_MANOPT_maxiter, 800)
            # set centers
            centers =  mid * Mdir[:RE] + Min[:RE] + im * (  mid * Mdir[:IM] + Min[:IM] )
            M = TROptModel(nrank1, dims, length(dims), Mdir[:RE] + im * Mdir[:IM], centers, Min[:RE] + im * Min[:IM],  lambda)
            # | StopBSEarly(sumopt_bstol, sumopt_bsrel)
            pX = quasi_Newton(M, f, grad_f, pstart; debug=debuginfo, stopping_criterion=StopAfterIteration(iter) | StopWhenChangeLess(M, sumopt_convtol) | StopWhenStepsizeLess(sumopt_convtol) , retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())
            # check if the point is better
            proj, pen, y = state(pX, M.dirs, M.centers, M.in, M.nrank1, M.sumdim, M.dims, M.cdims, M.ndim)
            if pen < sumopt_outerconvtol
                if proj > bestt
                    bestpoint = deepcopy(y)
                    bestt = proj
                    bestsol = deepcopy(pX)
                end
                left = mid
            else
                right = mid
            end
        end
        centers =  bestt * Mdir[:RE] + Min[:RE] + im * ( bestt * Mdir[:IM] + Min[:IM] )
        pstart = bestsol
        lambda = 2000
        prevproj = bestt
        proj = bestt
        for i in 1:1
            iter = min(param.heur_MANOPT_maxiter, 60)
            M = TROptModel(nrank1, dims, length(dims), Mdir[:RE] + im * Mdir[:IM], centers, Min[:RE] + im * Min[:IM], lambda)
            pX = quasi_Newton(M, f, grad_f, pstart; debug=debuginfo, stopping_criterion=StopAfterIteration(iter) | StopWhenChangeLess(M, sumopt_polishtol) | StopWhenStepsizeLess(sumopt_polishtol) | StopPolishEarly(sumopt_polishrel), retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())
            proj_, _, _ = state(pX, M.dirs, M.centers, M.in, M.nrank1, M.sumdim, M.dims, M.cdims, M.ndim)
            if abs(proj_ - prevproj) < sumopt_ttol
                lambda *= 1.1
            end
            prevproj = proj
            proj += (proj_ - proj) * 2 / (i + 1)
            centers =  proj * Mdir[:RE] + Min[:RE] + im * (  proj * Mdir[:IM] + Min[:IM] )
            pstart = pX
        end
    end
    purestates, substates = getXvals(pX, nrank1, sumdim, dims, cdims, ndim)
    return purestates, substates
end
