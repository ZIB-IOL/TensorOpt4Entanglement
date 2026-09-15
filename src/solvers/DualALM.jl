struct DualLiftModel <: AbstractLiftModel
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

function log!(M::DualLiftModel, X, p, q)
    X .= q - p
    return X
end

function fviolate(M, p, maxrank1 = M.nrank1)
    y = liftMap(M, p, maxrank1)
    dirs = M.dirs
    inner = dot(real(y), real(dirs)) +  dot(imag(y), imag(dirs))
    violate = inner + 1
    return y, inner, violate
end

# NOTE: the rescaling this retraction was meant to apply is commented out
# upstream, so it is currently a plain translation. Kept as-is: enabling it
# changes the dual ALM trajectory.
function retract_project!(M::DualLiftModel, q, p, dp)
    q .= p + dp
    return q
end

function makeObjectiveClosures(M::DualLiftModel, dirs, Mout, indexmap)
    function func(M, p, maxrank1 = M.nrank1)
        y, inner, violate = fviolate(M, p, maxrank1)
        f = norm(y -Mout)^2 / 2 # dot(real(y), real(Mout)) + dot(imag(y), imag(Mout))
        return f, inner, violate
    end

    function fastgrad_l_closure(M, p)
        y, _, violate = fviolate(M, p)
        linearize = y - Mout
        vg = zeros(Float64, length(p))  # Initialize gradient vector to zero
        liftGradient!(M, vg, p, linearize, indexmap)
        return vg
    end

    function l_closure(M, p)
        f, _, _ = func(M, p)
        return f
    end

    return func, l_closure, fastgrad_l_closure, retract_project!
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

"""
    dualALMSolve(detector, dims, H, substates, weights, multipliers, param,
                 is_escaping = false, is_high_accuracy = false)

Experimental dual augmented-Lagrangian solve (`-a LDual`). Not part of the
paper.

Runs two projection solves on the lift -- one towards `H`, one towards its
reflection -- and returns the resulting factors.

!!! warning "Work in progress"
    The returned feasibility residual is always `0.0`: the penalty schedule this
    routine was meant to drive was never wired up. `multipliers` is likewise
    ignored (the dual is pinned at `multi_min`) and the quasi-Newton iteration
    budget is hard-coded to 250 rather than taken from `param`. Treat its
    output as indicative only.
"""
function dualALMSolve(detector, dims::Vector{Int64}, H, substates, weights, multipliers, param::Param, is_escaping = false, is_high_accuracy = false)
    step_tol = param.heur_LADMM_step_tol * (is_high_accuracy ? 0.1 : 1)
    multi_min = 10
    ηmax = 10
    maxmanoptiter = 250

    dimH = reduce(*, dims)
    nsubs = length(dims)
    sumdim = reduce(+, dims)
    cdims = cumulativeAdd!(deepcopy(dims))
    nrank1 = length(substates)
    @assert(length(weights) == nrank1)

    Min = Dict(:RE => Matrix(Diagonal(ones(dimH) / dimH)), :IM => zeros(dimH, dimH))
    Mdir = Dict(:RE => real(H) - Min[:RE], :IM => imag(H) - Min[:IM])

    pX0, nrank1 = packFactors(substates, weights, nrank1, sumdim, dims, cdims, nsubs)
    pX00 = copy(pX0)
    debuginfo = [:Iteration, (:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), (:Stepsize, " Stepsize: %1.11f | "), DebugDualLiftState(), "\n", :Stop]

    pX = pX0
    Mdir_c = Mdir[:RE] .+ im .* Mdir[:IM]
    Mout_c = H
    multipliers_c = multi_min

    M = DualLiftModel(nrank1, Mdir_c, dims, nsubs)
    cur_pen = 0.0
    indexmap = buildIndexMap(dims)

    # update functions
    func, l_closure, fastgrad_l_closure = makeObjectiveClosures(M, Mdir_c, Mout_c, indexmap)
    state = quasi_Newton(M, l_closure, fastgrad_l_closure, pX; max_step_size = ηmax, debug=debuginfo, return_state=true,  record=[:Iteration],
        stopping_criterion=StopAfterIteration(maxmanoptiter) | StopWhenChangeLess(M, step_tol) ,
        retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())

    pX = get_solver_return(state)
    ypos = liftMap(M, pX)
    inner_pos = dot(real(ypos), real(H)) + dot(imag(ypos), imag(H))

    Mout_new = Mout_c - ypos
    Mout_new = -Mout_new
    pX = pX00
    func, l_closure, fastgrad_l_closure = makeObjectiveClosures(M, Mdir_c, Mout_new, indexmap)
    state = quasi_Newton(M, l_closure, fastgrad_l_closure, pX; max_step_size = ηmax, debug=debuginfo, return_state=true,  record=[:Iteration],
        stopping_criterion=StopAfterIteration(maxmanoptiter) | StopWhenChangeLess(M, step_tol) ,
        retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())

    pX = get_solver_return(state)
    yneg = liftMap(M, pX)
    f, inner, violate = func(M, pX)
    inner_neg = dot(real(yneg), real(H)) + dot(imag(yneg), imag(H))
    println("after: f = $f, inner = $inner, $inner_pos, $inner_neg")

    purestates_, substates_, weights_ = unpackFactorsUnscaled(pX, nrank1, sumdim, dims, cdims, nsubs)
    return purestates_, substates_, cur_pen, weights_
end

