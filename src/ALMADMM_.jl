
using Zygote
using LinearAlgebra, Manifolds, ManifoldsBase
using RecursiveArrayTools
import ManifoldsBase: representation_size,  manifold_dimension, inner, zero_vector, zero_vector!, retract_project!, exp!, inverse_retract!, parallel_transport_to!, rand!, log!, rand, copy, allocate_result,  allocate_result_array
import Manopt: max_stepsize, get_reason


function p2Vars(p, nrank1, dims, cdims, sumdim, nsubs)
    y = 0
    yp = 0
    weights = p.x[2]
    mat_p = p.x[1]
    d = dims[1]
    for i in 1:nrank1
        prod = 1
        xblock = mat_p[:, (i - 1) * nsubs + 1: i * nsubs]
        for j in 1:nsubs
            sphere = xblock[: , j]
            x = sphere[1:d] + im * sphere[d+ 1: 2 * d]
            xx = x * x'
            prod = kron(prod, xx)
        end
        weight = weights[i]
        if i == 1
            y = prod * weight
            yp = prod
        else
            y += prod * weight
            yp += prod
        end
    end
    print(norm(yp), " ", sum(weights), "\n")
    return y, p.x[3][1]
end


function flat_p2Vars(flat_p, nrank1, dims, cdims, sumdim, nsubs)
    y = 0
    d = dims[1]
    weights = flat_p[nrank1 * nsubs * 2 * d + 1 :  nrank1 * nsubs * 2 * d + nrank1]
    for i in 1:nrank1
        prod = 1
        xblock = flat_p[(i - 1) * nsubs * 2 * d + 1 : i * nsubs * 2 * d]
        for j in 1:nsubs
            sphere = xblock[ (j -1) * 2 * d  + 1 : j * 2 *d ]
            x = sphere[1:d] + im * sphere[d + 1: 2 * d]
            xx = x * x'
            prod = kron(prod, xx)
        end
        weight = weights[i]
        prod *= weight
        if i == 1
            y = prod
        else
            y += prod
        end
    end
    return y, flat_p[end]
end

function flatten_point(p)
    return vcat(p...)
end

function flatten_point_with_shape(p)
    flat = vcat(p...)
    shapes = map(size, p.x)
    return flat, shapes
end


function unflatten_point(flat::Vector{Float64}, shapes::Tuple)
    parts = ()
    idx = 1
    for sh in shapes
        len = prod(sh)
        slice = flat[idx : idx + len - 1]
        part = length(sh) == 1 ? slice : reshape(slice, sh)
        parts = (parts..., part)
        idx += len
    end
    return ArrayPartition(parts)
end

function make_objective_closures(M, dirs, Min, nrank1, dims, cdims, sumdim, nsubs, multipliers, lambda)
    # Helper: flatten manifold point to vector for AD

    function func(p)
        y, z = p2Vars(p, nrank1, dims, cdims, sumdim, nsubs)
        Aza = dirs * z + Min
        violate = y - Aza
        pen = dot(real(violate), real(violate)) + dot(imag(violate), imag(violate))
        f = -z
        L = dot(real(multipliers), real(violate)) + dot(imag(multipliers), imag(violate))
        return f,L,pen,violate
    end

    function f_closure(M, p)
        y, z = p2Vars(p, nrank1, dims, cdims, sumdim,  nsubs)
        #print(size(z), " ", size(dirs))
        Aza = dirs * z
        Aza += Min
        violate = y - Aza
        if isnan(norm(y))
            abort()
        end
        print(norm(y),"norm\n")
        pen = dot(real(violate), real(violate)) + dot(imag(violate), imag(violate))
        f = -z
        L = dot(real(multipliers), real(violate)) + dot(imag(multipliers), imag(violate))
        return f + L + lambda * pen
    end


    function f_closure_flat(flat_p)
        # Flatten the point p to a vector for AD
        y, z = flat_p2Vars(flat_p, nrank1, dims, cdims, sumdim, nsubs)
        Aza = dirs * z + Min
        violate = y - Aza
        pen = dot(real(violate), real(violate)) + dot(imag(violate), imag(violate))
        f = -z
        L = dot(real(multipliers), real(violate)) + dot(imag(multipliers), imag(violate))
        return f + L + lambda * pen
    end

    function grad_f_closure(M, p)
        flat_p, shapes = flatten_point_with_shape(p)
        vg = gradient(flat_p -> f_closure_flat(flat_p), flat_p)[1]
        gp = map(x -> zero(x), p)
        return gp
    end

    return f_closure, grad_f_closure, func
end


function getXvals(p, nrank1, dims, nsubs)
    Xvalss = []
    Xsubs = []
    for i in 1:nrank1
        prod = 1
        xs = []
        for j in 1:nsubs
            sphere = p[(i - 1) * nsubs + j]
            dim = dims[j]
            x = sphere[1:dim] + im * sphere[dim + 1: 2 * dim]
            xx = x * x'
            prod = kron(prod, xx)
            push!(xs, x)
        end
        @assert( abs(tr(prod) - 1) < 1e-6 )
        push!(Xvalss, prod)
        push!(Xsubs, xs)
    end
    return Xvals, Xsubs
end

function max_stepsize(M::ProductManifold)
    return 0.1
end

function ALMADMMSolve(dims::Vector{Int64}, H, substates, weights, z, multipliers, param::Param)
    #Test()
    sumopt_ttol = 1e-4
    sumopt_bsrel = 1e-2
    sumopt_bstol = 1e-3
    sumopt_convtol = 5e-5
    sumopt_polishtol = 1e-7
    sumopt_polishrel = 1e-3
    sumopt_outerconvtol = 5e-5
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
    d1 = dims[1]
    for d in dims
        @assert d == d1
    end

    @assert(length(weights) == nrank1)
    c = 1
    states_mat = zeros( 2 * d1, nsubs * nrank1)
    for i in 1:nrank1
        for sys in 1:nsubs
            states_mat[1:d1, c] =  real(substates[i][sys])
            states_mat[d1 + 1:2*d1, c] =  imag(substates[i][sys])
            c += 1
        end
    end
    pX0 = ArrayPartition((states_mat, weights, [z])...)
    #print(f(M, pX0), " ", dot(real(Hbar), real(H)) + dot(imag(Hbar), imag(H)))
    #stateentry = DebugEntry(state; format = "state %1.11f %1.11f", io=stdout)
    debuginfo = [:Iteration, (:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), (:Stepsize, " Stepsize: %1.11f | "), "\n", :Stop]
    debuginfo2 = [:Iteration, (:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), "\n", :Stop]

    # Bisection line search for maximum point
    lambda_min = 1e-2
    lambda = 1e-3
    lambdascale = 2.0

    pX = pX0
    # Outer power manifold
    Tsphere = PowerManifold(Sphere(2 * d1 - 1), nrank1 * nsubs)
    # Other components
    simplex = ProbabilitySimplex(nrank1 - 1; boundary = :open)
    positiveorthant = Euclidean(1)

    # Final manifold
    M = ProductManifold(Tsphere, simplex, positiveorthant)
    print(size(pX0[1]), " ", size(pX[1]), size(rand(simplex)), " ",  size(rand(Tsphere)), " ", size(states_mat), " ", size(rand(positiveorthant)),   " ", length(weights),"\n")
    for _ in 1:20  # adjust number of iterations as needed
        # set start point
        iter = min(param.heur_MANOPT_maxiter, 800)
        # set centers
        # | StopBSEarly(sumopt_bstol, sumopt_bsrel)
        myf, mygrad_f, func = make_objective_closures(M, Mdir[:RE] + im * Mdir[:IM], Min[:RE] + im * Min[:IM], nrank1, dims, cdims, sumdim, nsubs, multipliers[:RE] + multipliers[:IM], lambda)
        pX = quasi_Newton(M, myf, mygrad_f, pX; debug=debuginfo )
        f, L, pen, violate = func(pX)
        # update multipliers
        M.multipliers += lambda * violate
        mygrad_f(M, p)
        if  sqrt(pen) > 1.0
            M.lambda = lambdascale * M.lambda
        else
            M.lambda = max( M.lambda / lambdascale, ambda_min)
        end
    end

    purestates, substates = getXvals(pX, nrank1, dims, nsubs)
    return purestates, substates
end
