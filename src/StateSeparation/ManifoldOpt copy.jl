using TensorOperations
using LinearAlgebra, Manifolds, ManifoldsBase
using Manopt
import ManifoldsBase: representation_size, retract_fused!,  manifold_dimension, inner, zero_vector, zero_vector!, retract_project!, retract!, inverse_retract!, parallel_transport_to!, rand!, log!
import Manopt: max_stepsize

function arrayVectorize(X, dims::Vector{Int64}, ndim::Int64)
    p = vcat(vec.(X)...)
    return p
end

function arrayDeVectorize(p, dims::Vector{Int64}, ndim::Int64)
    arrayends = [dim * dim for dim in dims]
    cumulativeAdd!(arrayends)
    X = [reshape(p[ (i == 1 ? 1 : arrayends[i - 1] + 1) : arrayends[i]], dim, dim) for (i, dim) in enumerate(dims)]
    return X
end

struct ManifoldOptModel <: AbstractManifold{ℂ}
    dims::Vector{Int64}
    ndim::Int64
    H
    rH
    dimH::Int64

    function ManifoldOptModel(dims, ndim, H, dimH)
        rH = tensorProductResahpe(-H, dims, ndim)
        new(dims, ndim, -H, rH, dimH)
    end
end

function manifold_dimension(M::ManifoldOptModel)
    return reduce(+, [dim * dim for dim in M.dims])
end

function representation_size(M::ManifoldOptModel)
    return (manifold_dimension(M), )
end

function zero_vector(M::ManifoldOptModel, p)
    return zeros(Complex{Float64}, representation_size(M))
end

function zero_vector!(M::ManifoldOptModel, X, p)
    fill!(X, 0 + 0 * im)
end

function rand!(M::ManifoldOptModel, X; vector_at = nothing)
    fill!(X, 0 + 0 * im)
end

function inverse_retract!(M::ManifoldOptModel, X, p, q)
    X.= q - p
end

function log!(M::ManifoldOptModel, X, p, q)
    X.= q - p
end

function parallel_transport_to!(M::ManifoldOptModel, Y, p, X, q)
    Y .= X
end

function retract_project!(M::ManifoldOptModel, q, p, dp)
    ndim = M.ndim
    p_ = p + dp
    X = arrayDeVectorize(p_, M.dims, M.ndim)
    for i in 1:ndim
        X[i] = projectDensityMat(X[i])
    end
    X = arrayVectorize(X, M.dims, M.ndim)
    q.= X
    return q
end


function inner(M::ManifoldOptModel, p, pX, pY)
    return dot(real(pX), real(pY)) + dot(imag(pX), imag(pY))
end

function max_stepsize(M::ManifoldOptModel)
    return 1
end

function tensorProductResahpe(H, dims::Vector{Int64}, ndim::Int64)
    reshape_dims = [dim for dim in reverse(dims)]
    reshape_dims = vcat(reshape_dims, reshape_dims)
    rH = reshape(H, reshape_dims...)
    permuindex = [i for i in 1 : 2 * ndim]
    for i in 1: ndim
        permuindex[2 *i - 1 ] = ndim + 1 - i
        permuindex[2 *i ] = ndim + 1 - i + ndim
    end
    rH = permutedims(rH, permuindex)
    return rH
end


# Objective function: Frobenius norm of the difference
function f(M::ManifoldOptModel, p)
    # Compute the tensor product of the matrices A_1, ..., A_n
    X = arrayDeVectorize(p, M.dims, M.ndim)
    Hbar = foldl(kron, X)
    #print(dot(real(Hbar), real(M.H)) + dot(imag(Hbar), imag(M.H)), " ")
    return dot(real(Hbar), real(M.H)) + dot(imag(Hbar), imag(M.H))
end

function gfi(X, rH, dims, ndim, i)
    b = TensorOperations.BaseCopy()

    IrH = collect(j for j in 1 : 2 * ndim)
    Irprod_except = Tuple(collect([j for j in 1 : 2 * (ndim - 1) ] ))
    Iempty = ntuple(_ -> 0, 0)
    Iout = Tuple([1, 2])

    prod_except_i = foldl(kron, [X[j] for j in 1:ndim if j != i])
    except_i_dims = vcat(dims[1:i-1], dims[i+1:end])
    rprod_except_i = tensorProductResahpe(prod_except_i, except_i_dims, ndim - 1)

    # Compute the product of norms of all vectors excluding x_i
    #norm_prod_except_i = prod(norm(X[j])^2 for j in 1:ndim if j != i)

    # Gradient of the loss w.r.t. x_i
    #grad_i =  -norm_prod_except_i * x_i
    IrH_i = Tuple([2*i-1, 2 * i])
    IrH_except_i = Tuple(vcat(IrH[1 : 2 * i - 2], IrH[ 2 * i + 1 : end ]))

    realreal = tensorcontract(real(rH), ( IrH_i, IrH_except_i ), false, real(rprod_except_i), (Irprod_except, Iempty), false, ( Iout , Iempty), 1.0, b)
    realimag = tensorcontract(real(rH), ( IrH_i, IrH_except_i ), false, imag(rprod_except_i), (Irprod_except, Iempty), false, ( Iout , Iempty), 1.0, b)
    imagimag = tensorcontract(imag(rH), ( IrH_i, IrH_except_i ), false, imag(rprod_except_i), (Irprod_except, Iempty), false, ( Iout , Iempty), 1.0, b)
    imagreal = tensorcontract(imag(rH), ( IrH_i, IrH_except_i ), false, real(rprod_except_i), (Irprod_except, Iempty), false, ( Iout , Iempty), 1.0, b)
    grad_i = realreal - imagimag + im * (realimag + imagreal)
    return grad_i
end

# Gradient of the objective function
function grad_f(M::ManifoldOptModel, p)
    # Compute the gradient of f with respect to each X_i
    # Extract the vector x_i
    X = arrayDeVectorize(p, M.dims, M.ndim)
    dims = M.dims
    ndim = M.ndim
    rH = M.rH
    grad = []
    for i in 1:ndim
        # Compute the kron product of all vectors excluding x_i
        grad_i = gfi(X, rH, dims, ndim, i)
        push!(grad, grad_i)
    end
    vgrad = arrayVectorize(grad, M.dims, M.ndim)
    return vgrad
end


function ManOptSolve(dims::Vector{Int64}, ndim::Int64, H, dimH::Int64, X0, seed, param::Param)
    #Test()
    Xvals = nothing
    algo = quasi_Newton
    #Hbar = constructFullSol(X0)
    if isnothing(X0)
        X0 = initalDensityMats(dims, seed)
    end
    if X0 isa Dict
        X0 = [X0[:RE][i] + im * X0[:IM][i] for i in 1:length(X0[:RE])]
    end
    M = ManifoldOptModel(dims, ndim, H, dimH)
    pX0 = arrayVectorize(X0, dims, ndim)
    #print(f(M, pX0), " ", dot(real(Hbar), real(H)) + dot(imag(Hbar), imag(H)))
    debuginfo = [(:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), (:Stepsize, " Stepsize: %1.11f | "), "\n", :Stop]
    #pX = algo(M, f, grad_f, pX0; debug=[(:Cost, " F(x): %1.11f | "), (:GradientNorm,  " |Df(x)|: %1.11f | "), (:Stepsize, " Stepsize: %1.11f | "), "\n", :Stop],  stopping_criterion=StopAfterIteration(param.heur_MANOPT_maxiter) | StopWhenChangeLess(M, param.tol    ), retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())
    pX = algo(M, f, grad_f, pX0; stopping_criterion=StopAfterIteration(param.heur_MANOPT_maxiter) | StopWhenChangeLess(M, param.tol) | StopWhenStepsizeLess(param.tol), retraction_method = ProjectionRetraction(), vector_transport_method = ParallelTransport())
    X = arrayDeVectorize(pX, dims, ndim)
    Xvals = Dict(:RE => [real(Xval) for Xval in X], :IM => [imag(Xval) for Xval in X])
    return Xvals, pX
end

