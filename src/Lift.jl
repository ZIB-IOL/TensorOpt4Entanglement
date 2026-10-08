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
representation_size(M::AbstractLiftModel) = (M.sumdim * M.nrank1 * 2,)

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
    rank = min(M.nrank1, maxrank1)
    rank <= 0 && return 0
    # Batch the rank-one outer products as V * V'. Keep this path free of
    # mutation so the public lifting map remains differentiable by Zygote.
    vectors = map(1:rank) do r
        block = (r - 1) * 2 * M.sumdim
        factors = map(1:M.nsubs) do k
            d = M.dims[k]
            offset = block + 2 * (M.cdims[k] - d)
            complex.(view(p, offset + 1:offset + d),
                     view(p, offset + d + 1:offset + 2d))
        end
        reduce(kron, factors)
    end
    V = reduce(hcat, vectors)
    return V * V'
end

struct LiftGradientWorkspace
    factors::Matrix{ComplexF64}
    weighted_factors::Matrix{ComplexF64}
    coefficient::Matrix{ComplexF64}
    prefixes::Matrix{ComplexF64}
    suffixes::Matrix{ComplexF64}
end

function LiftGradientWorkspace(M::AbstractLiftModel)
    D = prod(M.dims)
    return LiftGradientWorkspace(
        zeros(ComplexF64, D, M.nrank1), zeros(ComplexF64, D, M.nrank1),
        zeros(ComplexF64, D, D), zeros(ComplexF64, D, M.nsubs + 1),
        zeros(ComplexF64, D, M.nsubs + 1))
end

function liftPrefixes!(M, work, p, r)
    prefix = work.prefixes
    prefix[1, 1] = 1
    leftdim = 1
    block = (r - 1) * 2 * M.sumdim
    @inbounds for k in 1:M.nsubs
        d = M.dims[k]
        offset = block + 2 * (M.cdims[k] - d)
        for a in 1:d, left in 1:leftdim
            prefix[(left - 1) * d + a, k + 1] =
                prefix[left, k] * complex(p[offset + a], p[offset + d + a])
        end
        leftdim *= d
    end
    return prefix
end

function liftFactors!(M, work, p)
    D = size(work.factors, 1)
    @inbounds for r in 1:M.nrank1
        liftPrefixes!(M, work, p, r)
        for i in 1:D
            work.factors[i, r] = work.prefixes[i, M.nsubs + 1]
        end
    end
    return work.factors
end

"""
    liftGradient!(M, vg, p, coefs, indexmap, workspace = LiftGradientWorkspace(M))

Euclidean gradient of `p -> real<coefs, liftMap(M, p)>`, written into `vg`.
For each product vector `v`, multiply by `(coefs + coefs')`, then contract
with the conjugate of the other mode vectors. A single matrix multiplication
handles all terms: O(r * (D^2 + m * D)), instead of O(r * m * D^2).
Prefix/suffix products handle zero factor entries. `indexmap` is retained for
compatibility with callers of the previous entrywise implementation.
"""
function liftGradient!(M, vg, p, coefs, indexmap,
                       work::LiftGradientWorkspace = LiftGradientWorkspace(M);
                       factors_ready = false)
    D = prod(M.dims)
    factors_ready || liftFactors!(M, work, p)
    # real<coefs, vv'> depends only on the Hermitian part of coefs.
    @inbounds for j in 1:D, i in 1:D
        work.coefficient[i, j] = coefs[i, j] + conj(coefs[j, i])
    end
    mul!(work.weighted_factors, work.coefficient, work.factors)

    fill!(vg, 0.0)
    @inbounds for r in 1:M.nrank1
        liftPrefixes!(M, work, p, r)
        block = (r - 1) * 2 * M.sumdim
        rightdim = 1
        work.suffixes[1, M.nsubs + 1] = 1
        for k in M.nsubs:-1:1
            d = M.dims[k]
            offset = block + 2 * (M.cdims[k] - d)
            for right in 1:rightdim, a in 1:d
                work.suffixes[(a - 1) * rightdim + right, k] =
                    complex(p[offset + a], p[offset + d + a]) * work.suffixes[right, k + 1]
            end
            rightdim *= d
        end
        leftdim = 1
        for k in 1:M.nsubs
            d = M.dims[k]
            rightdim = D ÷ (leftdim * d)
            offset = block + 2 * (M.cdims[k] - d)
            for a in 1:d
                g = zero(ComplexF64)
                for right in 1:rightdim, left in 1:leftdim
                    i = ((left - 1) * d + a - 1) * rightdim + right
                    g += conj(work.prefixes[left, k] * work.suffixes[right, k + 1]) *
                         work.weighted_factors[i, r]
                end
                vg[offset + a] = real(g)
                vg[offset + d + a] = imag(g)
            end
            leftdim *= d
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
    p = Float64[]
    sizehint!(p, 2 * sumdim * nrank1)
    total_trace = 0.0
    kept = 0
    for i in 1:nrank1
        weight = weights[i]
        isfinite(weight) && weight >= 0 || throw(DomainError(weight,"Weights must be nonnegative and finite"))
        weight < 1e-9 && continue
        scale = weight^(1 / (2 * nsubs))
        term_trace = weight
        for j in 1:nsubs
            v = substates[i][j]
            length(v) == dims[j] || throw(DimensionMismatch("Local factor has the wrong dimension"))
            term_trace *= sum(abs2, v)
            append!(p, scale .* real.(v))
            append!(p, scale .* imag.(v))
        end
        total_trace += term_trace
        kept += 1
    end
    # tr(kron(v₁v₁', …, vₘvₘ')) = prod(norm(vₖ)^2); checking it needs no
    # density matrices or repeated tensor products.
    @assert abs(total_trace - 1) < 1e-6
    return p, kept
end

"""
    unpackFactors(p, nrank1, sumdim, dims, cdims, nsubs)

Inverse of [`packFactors`](@ref): split a lift point back into normalised
rank-one density tensors, their unit-norm mode vectors, and the convex weights.
Assumes `p` has already been rescaled to unit total trace.
"""
function unpackFactors(p, nrank1, sumdim, dims, cdims, nsubs)
    purestates, substates, weights = [], [], Float64[]
    total_trace = 0.0
    for i in 1:nrank1
        block = (i - 1) * 2 * sumdim
        factors = [complex.(view(p, block + 2 * (cdims[j] - dims[j]) + 1:
                                   block + 2 * (cdims[j] - dims[j]) + dims[j]),
                            view(p, block + 2 * (cdims[j] - dims[j]) + dims[j] + 1:
                                   block + 2 * cdims[j])) for j in 1:nsubs]
        norms_squared = [sum(abs2, v) for v in factors]
        weight = prod(norms_squared)
        # A zero component contributes nothing to Psi; avoid dividing it by 0.
        iszero(weight) && continue
        for j in 1:nsubs
            factors[j] ./= sqrt(norms_squared[j])
        end
        v = reduce(kron, factors)
        push!(purestates, v * v')
        push!(substates, factors)
        push!(weights, weight)
        total_trace += weight
    end
    @assert abs(total_trace - 1) < 1e-6
    return purestates, substates, weights
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
