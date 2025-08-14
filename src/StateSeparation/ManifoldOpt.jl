
using Zygote
using LinearAlgebra, Manifolds, ManifoldsBase
using RecursiveArrayTools
import ManifoldsBase: representation_size, manifold_dimension, inner, zero_vector, zero_vector!, retract_project!, parallel_transport_to!, log!, rand!, copy
import Manopt: max_stepsize, get_reason, get_solver_return


struct Spheremodel <: AbstractManifold{ℝ}
    dims::Vector{Int64}
    cdims::Vector{Int64}
    nsubs::Int64
    sumdim::Int64

    function Spheremodel(dims, nsubs)
        cdims = deepcopy(dims)
        cdims = cumulativeAdd!(cdims)
        new(dims, cdims, nsubs, reduce(+,dims))
    end
end



function manifold_dimension(M::Spheremodel)
    return M.sumdim * 2
end

function representation_size(M::Spheremodel)
    return (manifold_dimension(M), )
end

function zero_vector(M::Spheremodel, p)
    return zeros(Float64, representation_size(M))
end

function zero_vector!(M::Spheremodel, X, p)
    fill!(X, 0.)
end

function rand!(M::Spheremodel, X; vector_at = nothing)
    fill!(X, 0.)
end


function inner(M::Spheremodel, p, pX, pY)
    return dot(pX, pY)
end


function vec2Var(M::Spheremodel, p)
    dims = M.dims
    cdims = M.cdims
    nsubs = M.nsubs
    prod = 1.0
    for j in 1:nsubs
        dim = dims[j]
        x = p[(cdims[j] - dim) * 2 + 1 : cdims[j] * 2]
        x = x[1:dim] + im * x[dim + 1: 2*dim]
        prod = kron(prod, x)
    end
    prod = prod * prod'
    return prod
end

function fastprojtangent!(M::Spheremodel, rg, p, g)
    dims = M.dims
    cdims = M.cdims
    nsubs = M.nsubs
    for j in 1:nsubs
        dim = dims[j]
        idx_start = (cdims[j] - dim) * 2 + 1
        idx_end = cdims[j] * 2
        x = p[idx_start : idx_end]
        gx = g[idx_start : idx_end]
        rg[idx_start : idx_end] = gx - dot(gx, x) / dot(x, x) * x
    end
    return rg
end


function retract_project!(M::Spheremodel, q, p, dp)
    t = 1.0
    q .= p + t * dp
    dims = M.dims
    cdims = M.cdims
    nsubs = M.nsubs
    for j in 1:nsubs
        dim = dims[j]
        idx_start = (cdims[j] - dim) * 2 + 1
        idx_end = cdims[j] * 2
        x = p[idx_start : idx_end]
        q[idx_start : idx_end] = x / norm(x)
    end
    return q
end

function log!(M::Spheremodel, X, p, q)
    X .= q - p
    #projtangent!(M, X, p, X)
    fastprojtangent!(M, X, p, X)
    return X
end

function parallel_transport_to!(M::Spheremodel, Y, p, X, q)
    # Y is the output vector, X is the vector to be transported
    Y .= X #projtangent!(M, Y, q, X)
    return Y
end


function SphereSolve(dims::Vector{Int64}, ndim::Int64, H, dimH::Int64, X0, seed, param::Param)

    nsubs = length(dims)

    cdims = deepcopy(dims)
    cdims = cumulativeAdd!(cdims)
    nsubs = length(dims)

    Xvals = nothing
    algo = quasi_Newton
    #Hbar = constructFullSol(X0)

    if isnothing(X0)
        X0 = initalDensityMats(dims, seed)
    end

    if X0 isa Dict
        X0 = [X0[:RE][i] + im * X0[:IM][i] for i in 1:length(X0[:RE])]
    end

    pX0 = []
    for i in 1:nsubs
        eigvals, eigvecs = eigen(X0[i])
        # Extract the eigenvector corresponding to the maximum eigenvalue
        maxndex = argmax(real(eigvals))   # Index of the maximum eigenvalue
        x = eigvecs[:, maxndex]     # Corresponding eigenvector
        # normalize the eigenvector
        x = x / norm(x)
        append!(pX0, real(x))
        append!(pX0, imag(x))
    end

    M = Spheremodel(dims, nsubs)

    indexmap = get_indexmap(dims)

    function f(M, p)
        y = vec2Var(M, p)
        return -dot(real(y), real(H))-dot(imag(y), imag(H))
    end

    function grad_f(M, p)
        dims = M.dims
        D = prod(dims)
        nsubs = M.nsubs

        g = copy(p)  # Initialize gradient vector to zero

        indexlen = nsubs * 2
        forward_prods = Vector{Complex{Float64}}(undef, indexlen)
        backward_prods = Vector{Complex{Float64}}(undef, indexlen)
        for i in 1:D
            for j in 1:D
                # get tensor index mapped back to products
                indices = indexmap[i, j]
                coef = H[i, j]
                real_coef = -real(coef)
                imag_coef = -imag(coef)
                # get the correponding block in flattened tensor
                x = p
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
                    g[real_idx] += gxr
                    g[imag_idx] += gxi
                end
            end
        end
        return g
    end

    #print(f(M, pX0), " ", dot(real(Hbar), real(H)) + dot(imag(Hbar), imag(H)))
    debuginfo = [(:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), (:Stepsize, " Stepsize: %1.11f | "), "\n", :Stop]
    #pX = algo(M, f, grad_f, pX0; debug=[(:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), (:Stepsize, " Stepsize: %1.11f | "), "\n", :Stop],  stopping_criterion=StopAfterIteration(param.heur_MANOPT_maxiter) | StopWhenChangeLess(M, param.tol    ), retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())
    pX = algo(M, f, grad_f, pX0; stopping_criterion=StopAfterIteration(param.heur_MANOPT_maxiter) | StopWhenChangeLess(M, param.tol) | StopWhenStepsizeLess(param.tol), retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())
    X = []
    for j in 1:nsubs
        dim = dims[j]
        x = pX[(cdims[j] - dim) * 2 + 1 : cdims[j] * 2]
        x = x[1:dim] + im * x[dim + 1: 2*dim]
        push!(X, x * x')
    end
    Xvals = Dict(:RE => [real(Xval) for Xval in X], :IM => [imag(Xval) for Xval in X])
    return Xvals, pX
end
