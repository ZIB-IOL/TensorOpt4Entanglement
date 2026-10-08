

# ---------------------------------------------------------------------------
# LADMM: lifted alternating direction method of multipliers (paper Alg. LADMM).
#
# Solves      min <c, y>   s.t.  y in SOST(d),  z in Z,  A(z) + a = y
# by replacing y with the smooth lift Psi(x) (see Lift.jl) and alternating:
#   x <- local min of the augmented Lagrangian over the manifold (quasi-Newton)
#   z <- exact minimiser (here a univariate quadratic on [0, 1])
#   chi <- chi + 2*zeta * (Psi(x) - A(z) - a)       (multiplier update)
#   zeta <- adapted from residual/gradient ratio, or by residual balancing
# The penalty is zeta*||residual||^2, so the usual ADMM penalty is rho = 2*zeta.
#
# For the white-noise mixing threshold the code parameterises
#   A(z) + a  ==  Mdir * z + Min,   Min = I/d-bar,  Mdir = phi - Min,
# which is the paper's parameterisation with z replaced by 1 - z; callers pass
# `1 - ub` in and read `1 - z` back out.
# ---------------------------------------------------------------------------

mutable struct LiftProjectionWorkspace
    point::Vector{Float64}
    normal::Vector{Float64}
    norms_squared::Matrix{Float64}
    term_products::Vector{Float64}
    normal_squared::Float64
    valid::Bool
end

struct LiftModel <: AbstractLiftModel
    nrank1::Int64
    dims::Vector{Int64}
    cdims::Vector{Int64}
    nsubs::Int64
    sumdim::Int64
    projection::LiftProjectionWorkspace

    function LiftModel(nrank1, dims, nsubs)
        cdims = deepcopy(dims)
        cdims = cumulativeAdd!(cdims)
        sumdim = sum(dims)
        projection = LiftProjectionWorkspace(zeros(2sumdim*nrank1),zeros(2sumdim*nrank1),
            zeros(nrank1,nsubs),zeros(nrank1),0.0,false)
        new(nrank1, dims, cdims, nsubs, sumdim, projection)
    end
end

# The normalised lift is a regular codimension-one level set of fastTrace.
manifold_dimension(M::LiftModel) = M.sumdim * M.nrank1 * 2 - 1
ManifoldsBase.project!(M::LiftModel, Y, p, X) = fastProjTangent!(M, Y, p, X)
ManifoldsBase.vector_transport_to_project!(M::LiftModel, Y, p, X, q; kwargs...) =
    fastProjTangent!(M, Y, q, X)

"""Stop an inner Manopt solve at the enclosing LADMM wall-clock deadline."""
mutable struct LADMMDeadline <: StoppingCriterion
    deadline::Float64
    reached::Bool
    LADMMDeadline(deadline) = new(deadline, false)
end

function (stop::LADMMDeadline)(::AbstractManoptProblem, ::AbstractManoptSolverState, ::Int)
    stop.reached = time() >= stop.deadline
    return stop.reached
end

get_reason(stop::LADMMDeadline) = stop.reached ? "LADMM wall-clock budget reached.\n" : ""

function ladmmLineSearch(M, p)
    # `max_step_size` is not a quasi_Newton keyword. Configure the line search
    # explicitly, including a positive bisection tolerance to avoid stagnation
    # when the two trial steps round to adjacent floating-point numbers.
    return WolfePowellLinesearch(M; p=copy(p), X=zero_vector(M, p),
        max_stepsize=0.1, sufficient_curvature=0.999,
        stop_when_stepsize_less=1e-10,
        retraction_method=ProjectionRetraction(), vector_transport_method=ProjectionTransport())
end

function ladmmPenalty(zeta, residual, dual_residual, grad_norm, policy)
    if policy == :legacy
        return residual > 0.8 * grad_norm ? min(zeta / 0.4, 200.0) : max(zeta * 0.4, 0.1)
    elseif policy != :balance
        throw(ArgumentError("LADMM penalty update must be :balance or :legacy"))
    end
    # Keep the penalty unchanged when the two residuals are comparable.
    residual > 10 * dual_residual && return min(2 * zeta, 200.0)
    dual_residual > 10 * residual && return max(zeta / 2, 0.1)
    return zeta
end

function vecNormSquare(M::LiftModel, p,
                       normsquare = zeros(Float64,M.nrank1,M.nsubs),
                       normsquareprod = zeros(Float64,M.nrank1))
    nrank1 = M.nrank1
    sumdim = M.sumdim
    dims = M.dims
    cdims = M.cdims
    nsubs = M.nsubs
    for i in 1:nrank1
        prod = 1.0
        for j in 1:nsubs
            dim = dims[j]
            idx_start = (i - 1) * sumdim * 2 + (cdims[j] - dim) * 2 + 1
            idx_end = (i - 1) * sumdim * 2 + cdims[j] * 2
            normsquare[i, j] = sum(abs2, view(p, idx_start:idx_end))
            prod *= normsquare[i, j]
        end
        normsquareprod[i] = prod
    end
    return normsquare, normsquareprod
end

function fastTrace(M::LiftModel, p)
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
            xnorm2 = sum(abs2, view(p, idx_start:idx_end))
            prod *= xnorm2
        end
        trace += prod
    end
    return trace
end

function fastProjTangent!(M::LiftModel, rg, p, g)
    work = M.projection
    gtr = work.normal
    # All L-BFGS history vectors are transported to the same new point. Its
    # trace normal is independent of the vector and needs computing only once.
    # Compare contents, since Manopt also updates point arrays in place.
    if !work.valid || p != work.point
        normsquare, normsquareprod = vecNormSquare(M,p,work.norms_squared,work.term_products)
        for i in 1:M.nrank1
            for j in 1:M.nsubs
                dim = M.dims[j]
                idx_start = (i - 1) * M.sumdim * 2 + (M.cdims[j] - dim) * 2 + 1
                idx_end = (i - 1) * M.sumdim * 2 + M.cdims[j] * 2
                # A zero factor has a zero derivative; otherwise use the exact
                # product of the other squared norms, without regularisation.
                coefficient = normsquare[i,j] > 0 ? 2 * normsquareprod[i] / normsquare[i,j] : 0.0
                @inbounds for a in idx_start:idx_end
                    gtr[a] = p[a] * coefficient
                end
            end
        end
        copyto!(work.point,p)
        work.normal_squared = dot(gtr,gtr)
        work.valid = true
    end
    denom = work.normal_squared
    denom > 0 || throw(DomainError(denom,"The trace level must be regular"))
    scale = dot(gtr, g) / denom
    rg .= g .- scale .* gtr
    return rg
end

#function exp!(M::LiftModel, q, p, dp, t::Float64)
#    q .= p + t * dp
#    trc = trace(M, q)
#    q /= trc^(1/(2*M.nsubs))
#    return q
function retract_project!(M::LiftModel, q, p, dp)
    t = 1.0
    q .= p + t * dp
    trc = fastTrace(M, q)
    q ./= trc^(1/(2*M.nsubs))
    return q
end

function log!(M::LiftModel, X, p, q)
    X .= q - p
    fastProjTangent!(M, X, p, X)
    return X
end

function makeObjectiveClosures(M::LiftModel, dirs, Min, multipliers, zeta, z, indexmap;
                               workspace = LiftGradientWorkspace(M), cached = false)
    # Manopt evaluates cost and gradient at the same point. Reuse the product
    # vectors and residual there; a copied point detects in-place trial updates.
    # The default path remains pure so AD can independently check the gradient.
    saved_point = cached ? Vector{Float64}(undef, 2 * M.sumdim * M.nrank1) : Float64[]
    saved_residual = cached ? similar(workspace.coefficient) : nothing
    cache_valid = false
    # Helper: flatten manifold point to vector for AD
    function fviolate(M, p, maxrank1 = M.nrank1)
        if cached && maxrank1 == M.nrank1
            if !cache_valid || p != saved_point
                V = liftFactors!(M, workspace, p)
                mul!(saved_residual, V, V')
                saved_residual .-= dirs .* z .+ Min
                copyto!(saved_point, p)
                cache_valid = true
            end
            return saved_residual
        end
        y = liftMap(M, p, maxrank1)
        #y /= (tr(y) + 1e-6)
        Aza = dirs * z + Min
        # y - dirs * z - Min
        violate = y - Aza
        return violate
    end

    function func(M, p, maxrank1 = M.nrank1)
        violate = fviolate(M, p, maxrank1)
        pen = real(dot(violate, violate))
        f = -z
        L = real(dot(multipliers, violate))
        return f, L, pen, violate
    end

    function al_closure(M, p, maxrank1 = M.nrank1)
        f, L, pen, _ = func(M, p, maxrank1)
        return  f + L + zeta * pen
    end


    function fastgrad_al_closure(M, p)

        violate = fviolate(M, p)
        linearize = multipliers + 2 * zeta * violate
        vg = zeros(Float64, length(p))  # Initialize gradient vector to zero
        liftGradient!(M, vg, p, linearize, indexmap, workspace; factors_ready=cached)
        fastProjTangent!(M, vg, p, vg)
        return vg
    end

    function fastgrad_l_closure(M, p)
        cached && fviolate(M, p)
        linearize = multipliers
        vg = zeros(Float64, length(p))  # Initialize gradient vector to zero
        liftGradient!(M, vg, p, linearize, indexmap, workspace; factors_ready=cached)
        fastProjTangent!(M, vg, p, vg)
        return vg
    end


    return al_closure, fastgrad_al_closure, func, fastgrad_l_closure
end

mutable struct LADMMColumnHistory
    point::Vector{Float64}
    scores::Vector{Float64}
    witness::Matrix{ComplexF64}
end

"Score unit product states under the preceding CP witness, independently of their weights."
function ladmmColumnScores(M, work, p, witness)
    V = liftFactors!(M, work, p)
    mul!(work.weighted_factors, witness, V)
    return map(1:M.nrank1) do r
        v, Mv = view(V, :, r), view(work.weighted_factors, :, r)
        weight = sum(abs2, v)
        weight > 1e-12 ? real(dot(v, Mv)) / weight : -Inf
    end
end

function recordLADMMColumns!(history, M, work, p)
    scores = ladmmColumnScores(M, work, p, history.witness)
    blocksize = 2M.sumdim
    for r in 1:M.nrank1
        if scores[r] > history.scores[r]
            rows = (r - 1) * blocksize + 1:r * blocksize
            copyto!(view(history.point, rows), view(p, rows))
            history.scores[r] = scores[r]
        end
    end
    return nothing
end

function appendLADMMColumns!(pool, history, M, work, p, offset, tol)
    final = ladmmColumnScores(M, work, p, history.witness)
    # Retain at most one earlier state per term. Each must violate the old CP
    # witness and score strictly better than that term's final state. Adding
    # these valid product columns can only enlarge the next inner approximation.
    keep = findall(r -> history.scores[r] > max(final[r], offset) + tol, 1:M.nrank1)
    isempty(keep) && return
    q = copy(history.point)
    q ./= fastTrace(M, q)^(1 / (2M.nsubs))
    P, S, _ = unpackFactors(q, M.nrank1, M.sumdim, M.dims, M.cdims, M.nsubs)
    append!(pool.pure, P[keep])
    append!(pool.sub, S[keep])
    return nothing
end

function ladmmIterationLimits(param, first_call, high_accuracy, dimH)
    outer = first_call ? param.heur_LADMM1_maxiter : param.heur_LADMM_maxiter
    inner = first_call ? param.heur_MANOPT1_maxiter : param.heur_MANOPT_maxiter
    inner = min(inner,2 * dimH^2 + 1)
    return high_accuracy ? (100000000,2 * inner) : (outer,inner)
end

"""
    ladmmSolve(detector, dims, H, substates, weights, z, multipliers, param,
                 is_escaping = false, is_high_accuracy = false)

LADMM: lifted alternating direction method of multipliers (paper Alg. LADMM).

Alternates a quasi-Newton local minimisation of the augmented Lagrangian over
the lift (`x`-step), the exact univariate minimisation over `z` (`z`-step), a
multiplier update, and a penalty update. The default `:legacy` policy uses the
original residual/gradient rule. With `param.heur_LADMM_penalty_update = :balance`,
the penalty doubles or halves only when the primal and dual residuals differ
by more than a factor of ten. Here the dual residual is
`2*zeta*norm(H - I/d)*abs(z - z_previous)`.
Inner solves share the enclosing wall-clock deadline.

When IR supplies `candidate_pool` and the preceding CP offset, retain earlier
product states that violate that witness more than the final states. The pool
adds columns to CP; it does not change the LADMM iterate or its residual.

Returns `(purestates, substates, 1 - z, residual, weights)`. The third value is
the heuristic upper bound `ub_heur`; it is only a valid bound on the original
problem when `residual` is zero to tolerance.
"""
ladmmSolve(detector, dims::Vector{Int64}, H, substates, weights, z, multipliers,
             param::Param, is_escaping = false, is_high_accuracy = false;
             candidate_pool = nothing, candidate_offset = -Inf) =
    withPhase(:ladmm) do
        ALMADMMSolve_(detector, dims, H, substates, weights, z, multipliers,
                      param, is_escaping, is_high_accuracy;
                      candidate_pool=candidate_pool, candidate_offset=candidate_offset)
    end

function ALMADMMSolve_(detector, dims::Vector{Int64}, H, substates, weights, z, multipliers, param::Param, is_escaping = false, is_high_accuracy = false;
    candidate_pool = nothing, candidate_offset = -Inf)
    obj_tol = param.heur_LADMM_obj_tol * ( is_high_accuracy ? 0.1 : 1)
    step_tol = param.heur_LADMM_step_tol * ( is_high_accuracy ? 0.1 : 1)
    gd_tol = param.heur_LADMM_gd_tol * ( is_high_accuracy ? 0.1 : 1)
    min_obj_tol = obj_tol * 0.1
    min_step_tol = step_tol * 0.1
    min_gd_tol = gd_tol * 0.1
    multi_bound = 1e2
    zeta = param.heur_LADMM_rho
    deadline = param.start_time + (param.is_last ? 1.0 : 1 - param.tratio) * param.time_limit

    dimH = reduce(*, dims)
    nsubs = length(dims)
    nrank1 = length(substates)
    # identity matrix
    Min = Dict(:RE=> Matrix( Diagonal(ones(dimH) / dimH)), :IM=>zeros(dimH, dimH))
    # direction matrix

    Mdir = Dict(:RE=> real(H) - Min[:RE], :IM=> imag(H) - Min[:IM])

    sumdim = reduce(+, dims)
    cdims = cumulativeAdd!(deepcopy(dims))

    @assert(length(weights) == nrank1)

    pX0, nrank1 = packFactors(substates, weights, nrank1, sumdim, dims, cdims, nsubs)
    debuginfo = param.log_level > 1 ?
        [:Iteration, (:Cost, " F(x): %1.11f | "), (:GradientNorm, " |Df(x)|: %1.11f | "), (:Stepsize, " Stepsize: %1.11f | "), "\n", :Stop] : []

    pX = pX0
    # Final manifold
    M = LiftModel(nrank1, dims, nsubs)

    Mdir_c = Mdir[:RE] .+ im .* Mdir[:IM]
    Min_c = Min[:RE] .+ im .* Min[:IM]
    multipliers_c = multipliers[:RE] .+ im .* multipliers[:IM]

    maxiter,maxmanoptiter = ladmmIterationLimits(param,is_escaping,is_high_accuracy,dimH)
    cur_pen = norm(liftMap(M, pX) - (Mdir_c * z + Min_c))
    direction_norm = norm(Mdir_c)
    indexmap = buildIndexMap(dims)
    workspace = LiftGradientWorkspace(M)
    history = isnothing(candidate_pool) || iszero(norm(multipliers_c)) ? nothing :
        LADMMColumnHistory(copy(pX), fill(-Inf, nrank1), -copy(multipliers_c))
    trace = ladmmTraceSink()

    i = 1
    while true  # adjust number of iterations as needed
        time() >= deadline && break

        # update manifold
        myf, mygrad_f, func, grad_l_closure = makeObjectiveClosures(M, Mdir_c, Min_c,
            multipliers_c, zeta, z, indexmap; workspace=workspace, cached=true)
        #y = liftMap(M, pX)
        #y /= tr(y)
        state = quasi_Newton(M, myf, mygrad_f, pX; debug=debuginfo, return_state=true, record=[:Iteration],
            memory_size=20, stepsize=ladmmLineSearch(M, pX),
            stopping_criterion=StopAfterIteration(maxmanoptiter) | StopWhenChangeLess(M, step_tol) | StopWhenCostChangeLess(obj_tol) | StopWhenGradientNormLess(gd_tol) | LADMMDeadline(deadline),
            project! = fastProjTangent!, retraction_method = ProjectionRetraction(),
            vector_transport_method = ProjectionTransport())
        pX = get_solver_return(state)

        iterations = get_record(state, :Iteration)
        last_iteration = isempty(iterations) ? 0 : last(iterations)
        # refine tolerance
        if (i == 1 || is_high_accuracy)  && last_iteration <= 3
            obj_tol = max( obj_tol / 2, min_obj_tol)
            step_tol = max( step_tol / 2, min_step_tol)
            gd_tol = max( gd_tol / 2, min_gd_tol)
        end

        # update z
        y = liftMap(M, pX)
        isnothing(history) || recordLADMMColumns!(history, M, workspace, pX)
        # a z^2 + b z + c
        y_in = y - Min_c
        a = 0
        b = -1
        b -= realInner(multipliers_c, Mdir_c)
        c = realInner(multipliers_c, y_in)
        a += realInner(Mdir_c, Mdir_c) * zeta
        c += realInner(y_in, y_in) * zeta
        b -= realInner(Mdir_c, y_in) * zeta * 2
        cstar, alstar = minimizeQuadraticOnUnitInterval(a, b, c)
        previous_z = z
        z = cstar

        # The x-step has not changed since y was computed above. Updating z
        # needs only this residual, rather than another lift and workspace.
        violate = y_in - Mdir_c * z
        f, L, pen = -z, real(dot(multipliers_c, violate)), real(dot(violate, violate))
        param.log_level > 1 && println("after: zeta = $zeta, f = $f, pen = $pen, alm = $(f + L + zeta * pen), z = $z")

        # update multipliers
        multipliers_c .+= (2 * zeta) .* violate
        # Clamp real and imaginary parts separately
        multipliers_c .= complex.(
            clamp.(real(multipliers_c), -multi_bound, multi_bound),
            clamp.(imag(multipliers_c), -multi_bound, multi_bound)
        )
        cur_pen = sqrt(pen)
        norm_vgl = norm(grad_l_closure(M, pX))
        # residual is ||A(z) + a - Psi(x)||_2: ub_heur is a valid bound only
        # once this vanishes, so its trajectory is what the paper discusses
        traceRow!(trace, i, zeta, f, pen, cur_pen, norm_vgl, z, f + L + zeta * pen)

        dual_residual = 2 * zeta * direction_norm * abs(z - previous_z)
        zeta = ladmmPenalty(zeta, cur_pen, dual_residual, norm_vgl, param.heur_LADMM_penalty_update)

        # check convergence
        feas_tol = obj_tol
        needbreak = false
        param.log_level > 1 && println("cur_pen: ", cur_pen, " < ", feas_tol, ", norm_vgl: ", norm_vgl, "<", min_gd_tol)
        if cur_pen < feas_tol && norm_vgl < min_gd_tol
            needbreak = true
        end

        # check time
        if time() >= deadline
            param.log_level > 0 && println("LADMM time budget reached, returning to CP...")
            needbreak = true
        end

        i += 1

        if i > maxiter
            needbreak = true
        end

        if needbreak
            break
        end

        if param.lazification && false
            purestates_, substates_, _ = unpackFactors(pX, nrank1, sumdim, dims, cdims, nsubs)
            addBatchStates(detector, purestates_, substates_, param.lazification)
        end
    end

    traceClose!(trace)
    trc = fastTrace(M, pX)
    pX /= trc^(1/(2*M.nsubs))
    isnothing(history) || appendLADMMColumns!(candidate_pool, history, M, workspace, pX,
        candidate_offset, param.master_obj_tol)
    purestates_, substates_, weights_ = unpackFactors(pX, nrank1, sumdim, dims, cdims, nsubs)
    return purestates_, substates_, 1 - z, cur_pen, weights_
end
