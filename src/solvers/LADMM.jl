

# ---------------------------------------------------------------------------
# LADMM: lifted alternating direction method of multipliers (paper Alg. LADMM).
#
# Solves      min <c, y>   s.t.  y in SOST(d),  z in Z,  A(z) + a = y
# by replacing y with the smooth lift Psi(x) (see Lift.jl) and alternating:
#   x <- local min of the augmented Lagrangian over the manifold (quasi-Newton)
#   z <- exact minimiser (here a univariate quadratic on [0, 1])
#   chi <- chi + zeta * (A(z) + a - Psi(x))         (multiplier update)
#   zeta <- adapted from the residual/gradient ratio
#
# For the white-noise mixing threshold the code parameterises
#   A(z) + a  ==  Mdir * z + Min,   Min = I/d-bar,  Mdir = phi - Min,
# which is the paper's parameterisation with z replaced by 1 - z; callers pass
# `1 - ub` in and read `1 - z` back out.
# ---------------------------------------------------------------------------

struct LiftModel <: AbstractLiftModel
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
            x = p[idx_start:idx_end]
            xnorm2 = dot(x, x)
            prod *= xnorm2
        end
        trace += prod
    end
    return trace
end

function fastProjTangent!(M::LiftModel, rg, p, g)

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

function makeObjectiveClosures(M::LiftModel, dirs, Min, multipliers, zeta, z, indexmap)
    # Helper: flatten manifold point to vector for AD
    function fviolate(M, p, maxrank1 = M.nrank1)
        y = liftMap(M, p, maxrank1)
        #y /= (tr(y) + 1e-6)
        Aza = dirs * z + Min
        # y - dirs * z - Min
        violate = y - Aza
        return violate
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
        return  f + L + zeta * pen
    end


    function fastgrad_al_closure(M, p)

        violate = fviolate(M, p)
        linearize = multipliers + 2 * zeta * violate
        vg = zeros(Float64, length(p))  # Initialize gradient vector to zero
        liftGradient!(M, vg, p, linearize, indexmap)
        fastProjTangent!(M, vg, p, vg)
        return vg
    end

    function fastgrad_l_closure(M, p)
        linearize = multipliers
        vg = zeros(Float64, length(p))  # Initialize gradient vector to zero
        liftGradient!(M, vg, p, linearize, indexmap)
        fastProjTangent!(M, vg, p, vg)
        return vg
    end


    function grad_l_closure(M, p)
        vg = gradient(p -> l_closure(M, p), p)[1]
        fastProjTangent!(M, vg, p, vg)
        return vg
    end

    return al_closure, fastgrad_al_closure, func, fastgrad_l_closure
end

"""
    ladmmSolve(detector, dims, H, substates, weights, z, multipliers, param,
                 is_escaping = false, is_high_accuracy = false)

LADMM: lifted alternating direction method of multipliers (paper Alg. LADMM).

Alternates a quasi-Newton local minimisation of the augmented Lagrangian over
the lift (`x`-step), the exact univariate minimisation over `z` (`z`-step), a
multiplier update, and a penalty update

    zeta <- zeta / 0.4   if  ||residual|| >  0.8 * ||dL/dx||
    zeta <- zeta * 0.4   otherwise

Returns `(purestates, substates, 1 - z, residual, weights)`. The third value is
the heuristic upper bound `ub_heur`; it is only a valid bound on the original
problem when `residual` is zero to tolerance.
"""
ladmmSolve(detector, dims::Vector{Int64}, H, substates, weights, z, multipliers,
             param::Param, is_escaping = false, is_high_accuracy = false) =
    withPhase(:ladmm) do
        ALMADMMSolve_(detector, dims, H, substates, weights, z, multipliers,
                      param, is_escaping, is_high_accuracy)
    end

function ALMADMMSolve_(detector, dims::Vector{Int64}, H, substates, weights, z, multipliers, param::Param, is_escaping = false, is_high_accuracy = false)
    obj_tol = param.heur_LADMM_obj_tol * ( is_high_accuracy ? 0.1 : 1)
    step_tol = param.heur_LADMM_step_tol * ( is_high_accuracy ? 0.1 : 1)
    gd_tol = param.heur_LADMM_gd_tol * ( is_high_accuracy ? 0.1 : 1)
    min_obj_tol = obj_tol * 0.1
    min_step_tol = step_tol * 0.1
    min_gd_tol = gd_tol * 0.1
    multi_bound = 1e2
    zeta_max = 2e2
    zeta_min = 1e-1
    zeta = param.heur_LADMM_rho
    zetascale = 0.4
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
    cdims = cumulativeAdd!(deepcopy(dims))

    @assert(length(weights) == nrank1)

    pX0, nrank1 = packFactors(substates, weights, nrank1, sumdim, dims, cdims, nsubs)
    debuginfo = [:Iteration, (:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), (:Stepsize, " Stepsize: %1.11f | "), "\n", :Stop]

    pX = pX0
    # Final manifold
    M = LiftModel(nrank1, dims, nsubs)

    Mdir_c = Mdir[:RE] .+ im .* Mdir[:IM]
    Min_c = Min[:RE] .+ im .* Min[:IM]
    multipliers_c = multipliers[:RE] .+ im .* multipliers[:IM]

    maxiter = is_escaping ? param.heur_LADMM_maxiter : param.heur_LADMM1_maxiter
    maxiter = is_high_accuracy ? 100000000 : maxiter
    cur_pen = 0.0
    indexmap = buildIndexMap(dims)
    trace = ladmmTraceSink()

    maxmanoptiter = min( is_escaping ? param.heur_MANOPT1_maxiter : param.heur_MANOPT_maxiter, dimH * dimH *2 +1)
    maxmanoptiter *= is_high_accuracy ? 2 : 1
    i = 1
    while true  # adjust number of iterations as needed

        # update manifold
        myf, mygrad_f, func, grad_l_closure = makeObjectiveClosures(M, Mdir_c, Min_c, multipliers_c, zeta, z, indexmap)
        #y = liftMap(M, pX)
        #y /= tr(y)
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

        # update z
        y = liftMap(M, pX)
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
        z = cstar

        myf, mygrad_f, func = makeObjectiveClosures(M, Mdir_c, Min_c, multipliers_c, zeta, z, indexmap)
        f, L, pen, violate = func(M, pX)
        println("after: zeta = $zeta, f = $f, pen = $pen, alm = $(f + L + zeta * pen), z = $z")

        # update multipliers
        multipliers_c += zeta * violate
        # Clamp real and imaginary parts separately
        multipliers_c = complex.(
            clamp.(real(multipliers_c), -multi_bound, multi_bound),
            clamp.(imag(multipliers_c), -multi_bound, multi_bound)
        )
        cur_pen = sqrt(pen)
        norm_vgl = norm(grad_l_closure(M, pX))
        # residual is ||A(z) + a - Psi(x)||_2: ub_heur is a valid bound only
        # once this vanishes, so its trajectory is what the paper discusses
        traceRow!(trace, i, zeta, f, pen, cur_pen, norm_vgl, z, f + L + zeta * pen)

        if cur_pen > norm_vgl * tau
            zeta = min(zeta / zetascale, zeta_max)
        else
            zeta = max(zeta * zetascale, zeta_min)
        end

        # check convergence
        feas_tol = obj_tol
        needbreak = false
        print("cur_pen: ", cur_pen, " < ", feas_tol, ", norm_vgl: ", norm_vgl, "<", min_gd_tol, "\n")
        if cur_pen < feas_tol && norm_vgl < min_gd_tol
            needbreak = true
        end

        # check time
        if isTimeLimitExceeded(param)
            println("Time limit exceeded, exiting...")
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
    purestates_, substates_, weights_ = unpackFactors(pX, nrank1, sumdim, dims, cdims, nsubs)
    return purestates_, substates_, 1 - z, cur_pen, weights_
end

