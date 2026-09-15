# ---------------------------------------------------------------------------
# The smooth lift (parameterisation) of the separable tensor cone.
#
# Paper notation -> code:
#   Psi   (lifting map)          -> `liftMap`
#   x     (point of the lift)    -> `p`, a flat Float64 vector
#   r     (factorisation size)   -> `nrank1`
#   m     (number of subsystems) -> `nsubs`
#   d-bar (prod of local dims)   -> `dimH`
#   d-til (sum  of local dims)   -> `sumdim`
#
# A point `p` packs `nrank1` blocks of length `2 * sumdim`. Block j holds the
# `nsubs` mode vectors of the j-th rank-one term, each stored as its `dim` real
# entries followed by its `dim` imaginary entries. Then
#     liftMap(M, p) = sum_j (x_j1 (x) ... (x) x_jm)(...)^dagger
# which is exactly Psi.
#
# `LiftModel` (solvers/LADMM.jl) and `DualLiftModel` (solvers/DualALM.jl) share
# this packing and therefore share everything below; they differ only in their
# retraction, logarithm and objective.
# ---------------------------------------------------------------------------

"""
    AbstractLiftModel

Manifolds whose points are the flat factorisation vectors described above.
Subtypes must provide the fields `nrank1`, `dims`, `cdims`, `nsubs`, `sumdim`.
"""
abstract type AbstractLiftModel <: AbstractManifold{ℝ} end

manifold_dimension(M::AbstractLiftModel)  = M.sumdim * M.nrank1 * 2
representation_size(M::AbstractLiftModel) = (manifold_dimension(M),)

zero_vector(M::AbstractLiftModel, p)      = zeros(Float64, representation_size(M))
zero_vector!(M::AbstractLiftModel, X, p)  = fill!(X, 0.0)
rand!(M::AbstractLiftModel, X; vector_at = nothing) = fill!(X, 0.0)

inner(M::AbstractLiftModel, p, pX, pY)    = dot(pX, pY)
max_stepsize(M::AbstractLiftModel)        = 0.1

function parallel_transport_to!(M::AbstractLiftModel, Y, p, X, q)
    Y .= X
    return Y
end

"""
    liftMap(M, p, maxrank1 = M.nrank1)

The lifting map Psi: sum the outer products of the first `maxrank1` rank-one
tensor factors packed in `p`.
"""
function liftMap(M::AbstractLiftModel, p, maxrank1 = M.nrank1)
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
            x = x[1:dim] + im * x[dim + 1: 2*dim]
            prod = kron(prod, x)
        end
        if i == 1
            y =  prod * prod'
        else
            y +=  prod * prod'
        end
    end
    return y
end

"""
    liftGradient!(M, vg, p, coefs, indexmap)

Euclidean gradient of `p -> real<coefs, liftMap(M, p)>`, accumulated into `vg`.

Computed analytically instead of by AD: for each tensor entry `(i, j)` the
product over modes is differentiated with the standard forward/backward prefix
trick, which is O(m) per entry rather than O(m^2).
"""
function liftGradient!(M, vg, p, coefs, indexmap)
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

"""
    packFactors(substates, weights, nrank1, sumdim, dims, cdims, nsubs)

Fold convex weights into the mode vectors and flatten them into a lift point:
scaling every mode vector of term j by `w_j^(1/(2m))` multiplies its rank-one
tensor by `w_j`, so `liftMap(M, p) == sum_j w_j * p_j`. Terms with negligible
weight are dropped; returns `(p, nrank1_kept)`.
"""
function packFactors(substates, weights, nrank1, sumdim, dims, cdims, nsubs)
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

"""
    unpackFactors(p, nrank1, sumdim, dims, cdims, nsubs)

Inverse of [`packFactors`](@ref): split a lift point back into normalised
rank-one density tensors, their unit-norm mode vectors, and the convex weights.
Assumes `p` has already been rescaled to unit total trace.
"""
function unpackFactors(p, nrank1, sumdim, dims, cdims, nsubs)
    Xvalss = []
    Xsubs = []
    weights = []
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
        push!(weights, real(trprod))
    end
    @assert( abs(sumtrprod - 1) < 1e-6)
    return Xvalss, Xsubs, weights
end

"""
    unpackFactorsUnscaled(p, nrank1, sumdim, dims, cdims, nsubs)

Like [`unpackFactors`](@ref) but performs no trace normalisation and reports
unit weights. Used by the dual ALM, whose iterates are not trace-normalised.
"""
function unpackFactorsUnscaled(p, nrank1, sumdim, dims, cdims, nsubs)
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
