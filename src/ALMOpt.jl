
using Zygote
using LinearAlgebra, Manifolds, ManifoldsBase
using Manopt
import ManifoldsBase: representation_size,  manifold_dimension, inner, zero_vector, zero_vector!, retract_project!, exp!, inverse_retract!, parallel_transport_to!, rand!, log!
import Manopt: max_stepsize, get_reason


struct ALMOptModel <: AbstractManifold{ℂ}
    nrank1::Int64
    dims::Vector{Int64}
    cdims::Vector{Int64}
    ndim::Int64
    sumdim::Int64
    dirs
    in
    lambdas
    rho

    function ALMOptModel(nrank1, dims, ndim, dirs, in, lambdas, rho)
        cdims = deepcopy(dims)
        cdims = cumulativeAdd!(cdims)
        new(nrank1, dims, cdims, ndim, reduce(+,dims), dirs, in, lambdas, rho)
    end
end

function manifold_dimension(M::ALMOptModel)
    return M.sumdim * M.nrank1
end

function representation_size(M::ALMOptModel)
    return (manifold_dimension(M), )
end

function zero_vector(M::ALMOptModel, p)
    return zeros(Complex{Float64}, representation_size(M))
end

function zero_vector!(M::ALMOptModel, X, p)
    fill!(X, 0 + 0 * im)
end

function rand!(M::ALMOptModel, X; vector_at = nothing)
    fill!(X, 0 + 0 * im)
end

function inverse_retract!(M::ALMOptModel, X, p, q)
    X.= q - p
end

function log!(M::ALMOptModel, X, p, q)
    X.= q - p
end

function parallel_transport_to!(M::ALMOptModel, Y, p, X, q)
    Y .= X
end


function inner(M::ALMOptModel, p, pX, pY)
    return dot(real(pX), real(pY)) + dot(imag(pX), imag(pY))
end

function max_stepsize(M::ALMOptModel)
    return 10
end

function retract_project!(M::ALMOptModel, q, p, dp)
    q .= p + dp
    trc = trace(q, M.nrank1, M.sumdim, M.dims, M.cdims, M.ndim)
    q /= trc^(1/(2*M.ndim))
    return q
end


function exp!(M::ALMOptModel, q, p, dp, t::Float64)
    q .= p + t * dp
    trc = trace(q, M.nrank1, M.sumdim, M.dims, M.cdims, M.ndim)
    q /= trc^(1/(2*M.ndim))
    return q
end


function almstate(xs, dirs, in, lambdas, rho, nrank1, sumdim, dims, cdims, nsubs)
    y = yy(xs, nrank1, sumdim, dims, cdims, nsubs)
    dirRE = real(dirs)
    dirIM = imag(dirs)
    yreal = real(y) - real(in)
    yimag = imag(y) - imag(in)
    obj = dot(yreal, dirRE) + dot(yimag, dirIM)
    slackRE = yreal.* dirRE - yreal
    slackIM = yimag.* dirIM - yimag
    augpen = dot(slackRE, slackRE) + dot(slackIM, slackIM)
    slackRE_ = slackRE + lambdas[:RE] / rho
    slackIM_ = slackIM + lambdas[:IM] / rho
    augpen_ = dot(slackRE_, slackRE_) + dot(slackIM_, slackIM_)
    return obj, slackRE, slackIM, augpen, augpen_
end

function almff(xs, dirs, in, lambdas, rho, nrank1, sumdim, dims, cdims, nsubs)
    obj, _, _, _, augpen_ = almstate(xs, dirs, in, lambdas, rho, nrank1, sumdim, dims, cdims, nsubs)
    return -obj + rho / 2 * augpen_
end


function almstate(M::ALMOptModel, p)
    # Compute the tensor product of the matrices A_1, ..., A_n
    obj, slackRE, slackIM, augpen, augpen_ = almstate(p, M.dirs, M.in, M.lambdas, M.rho, M.nrank1, M.sumdim, M.dims, M.cdims, M.ndim)
    return obj, slackRE, slackIM, augpen, augpen_
end


mutable struct DebugALMState{TIO<:IO} <: DebugAction
    io::TIO
    DebugALMState(io::IO=stdout) = new{typeof(io)}(io)
end

function (d::DebugALMState)(amp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    p = s.p
    M = get_manifold(amp)
    obj, _, _, augpen, _ = almstate(M, p)
    (k >= 0) && (print(d.io, (obj,augpen)))
    return nothing
end

mutable struct StopEarlyALM <: StoppingCriterion
    prevpen::Float64
    tol::Float64
    reltol::Float64
    fails::Int
    StopEarlyALM(tol::Float64, reltol::Float64) = new(Inf, tol, reltol, 0)
end

function get_reason(sc::StopEarlyALM)
    return "early stopping\n"
end

function (se::StopEarlyALM)(amp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    p = s.p
    M = get_manifold(amp)
    _, pen = almstate(M, p)
    fail = abs(se.prevpen - pen) < se.reltol * max(se.prevpen, pen) && pen > se.tol
    se.prevpen = pen
    if fail
        se.fails += 1
    end
    return se.fails >= 3
end



# Objective function: Frobenius norm of the difference
function f(M::ALMOptModel, p)
    # Compute the tensor product of the matrices A_1, ..., A_n
    return almff(p, M.dirs,  M.in, M.lambdas, M.rho, M.nrank1, M.sumdim, M.dims, M.cdims, M.ndim)
end

# Gradient of the objective function
function grad_f(M::ALMOptModel, p)
    # Compute the gradient of f with respect to each X_i
    # Extract the vector x_i
    vg = gradient(xs -> almff(xs, M.dirs,  M.in, M.lambdas, M.rho, M.nrank1, M.sumdim, M.dims,  M.cdims,  M.ndim), p)
    return vg[1]
end



function ALMOptSolve(dims::Vector{Int64}, H, param::Param)
    #Test()
    sumopt_ttol = 1e-4
    sumopt_bsrel = 1e-2
    sumopt_bstol = 1e-3
    sumopt_convtol = 5e-5
    sumopt_polishtol = 1e-7
    sumopt_polishrel = 1e-3
    sumopt_outerconvtol = 5e-5

    dimH = reduce(*, dims)

    epsilon=1e-3
    epsilon_min=1e-6
    lambdas0 = Dict(:RE=> ones(dimH, dimH), :IM=>ones(dimH, dimH))
    lambdas = deepcopy(lambdas0)
    lambda_max=20.0
    lambda_min=-lambda_max
    rho=1.0
    tau=0.8
    theta_rho=0.3
    epsilon_exponent=1 / 100
    theta_epsilon=(epsilon_min / epsilon)^(epsilon_exponent)


    ndim = length(dims)
    nrank1 = dimH * dimH
    Min = Dict(:RE=> Matrix( Diagonal(ones(dimH) / dimH)), :IM=>zeros(dimH, dimH))
    Mdir = Dict(:RE=>real(H) - Min[:RE], :IM=>imag(H) - Min[:IM])
    Mdirnorm = sqrt(dot(Mdir[:RE], Mdir[:RE]) + dot(Mdir[:IM], Mdir[:IM]))
    Mdir[:RE] /= Mdirnorm
    Mdir[:IM] /= Mdirnorm

    sumdim = reduce(+, dims)
    cdims = deepcopy(dims)
    cdims = cumulativeAdd!(cdims)
    feasiblestart = true
    if feasiblestart
        pX = TROptSolve(dims, H, true, param)
    else
        pX = randn(sumdim * nrank1) + im * randn(sumdim * nrank1)
        trc = trace(pX, nrank1, sumdim, dims, cdims, ndim)
        pX /= trc^(1/(2*ndim))
    end
    debuginfo = [:Iteration, (:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), (:Stepsize, " Stepsize: %1.11f | "), DebugALMState(), "\n", :Stop]
    _, slackRE, slackIM, _ = almstate(pX, Mdir[:RE] + im * Mdir[:IM], Min[:RE] + im * Min[:IM], lambdas, rho, nrank1, sumdim, dims, cdims, ndim)
    cost_eq = Dict(:RE=>slackRE, :IM=>slackIM)
    penalty_ = max(maximum(abs.(cost_eq[:RE])), maximum(abs.(cost_eq[:IM])))
    for i in 1:20
        iter = min(param.heur_MANOPT_maxiter, 30)
        M = ALMOptModel(nrank1, dims, length(dims), Mdir[:RE] + im * Mdir[:IM], Min[:RE] + im * Min[:IM], lambdas, rho)
        print((penalty, rho), "\n")
        pX = quasi_Newton(M, f, grad_f, pX; debug=debuginfo, stopping_criterion=StopAfterIteration(iter) | StopWhenChangeLess(M, sumopt_polishtol) | StopWhenStepsizeLess(sumopt_polishtol), retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())
        _, slackRE, slackIM, _, _ = almstate(pX, M.dirs, M.in,  M.lambdas, M.rho, M.nrank1, M.sumdim, M.dims, M.cdims, M.ndim)

        cost_eq = Dict(:RE=>slackRE, :IM=>slackIM)
        for part in (:RE,:IM)
            lambdas[part] .=
                min.(
                    lambdas0[part] .* lambda_max,
                    max.(lambdas0[part] .* lambda_min, lambdas[part] + rho .* cost_eq[part]),
                )
        end
        # get new evaluation of penalty
        penalty = max(maximum(abs.(cost_eq[:RE])), maximum(abs.(cost_eq[:IM])))
        # update rho if necessary
        (penalty > tau * penalty_) && (rho = rho / theta_rho)
        penalty_ = penalty

        # update the tolerance epsilon
        epsilon = max(epsilon_min, epsilon * theta_epsilon)
    end

    purestates = getXvals(pX, nrank1, sumdim, dims, cdims, ndim)
    return purestates
end
