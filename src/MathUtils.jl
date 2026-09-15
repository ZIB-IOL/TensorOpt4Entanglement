# ---------------------------------------------------------------------------
# Small numeric helpers shared across the algorithms.
# ---------------------------------------------------------------------------

function minimizeQuadraticOnUnitInterval(a, b, c)
    if abs(a) < 1e-12
        zstar = b > 0 ? 0.0 : 1.0
    else
        zstar = -b / (2a)
        zstar = clamp(zstar, 0.0, 1.0)
    end
    f0 = c
    f1 = a + b + c
    fz = a*zstar^2 + b*zstar + c
    if fz <= f0 && fz <= f1
        return zstar, fz
    elseif f0 <= f1
        return 0.0, f0
    else
        return 1.0, f1
    end
end

function realInner(Mdir_c, Min_c)
   inner_real = dot(real(Mdir_c), real(Min_c)) + dot(imag(Mdir_c), imag(Min_c))
   return inner_real
end

"""
    classifyStatus(model, status, primal_status, dual_status, primalobj, dualobj;
                   obj_tol, infeasible_tag, unknown_is_nosolution)

Map a JuMP termination status onto the internal `Status` enum.

Two call sites disagree on the details and the difference is deliberate:
`solveMSK` reports a plain `RelaxInfeasible`, while `solveAlternate` needs to
distinguish an infeasibility *certificate* (it prices off the dual ray) and
treats an unknown primal/dual status as "no solution".
"""
function cumulativeAdd!(c::Vector{T}) where T
    for i in 2:length(c)
        c[i] += c[i - 1]  # Add the previous element to the current element
    end
    return c
end

function unravelIndex(k, dims)
   idxs = []
   for d in reverse(dims)
      pushfirst!(idxs, mod(k, d)+1)
      k ÷= d
   end
   return idxs
end

function buildIndexMap(dims)
   D = prod(dims)
   indexmap = Matrix{Vector{Tuple{Int, Int, Int}}}(undef, D, D)
   for row in 1:D, col in 1:D
       indexmap[row, col] = Tuple{Int, Int, Int}[]
   end
   cdims = cumulativeAdd!(deepcopy(dims))
   for i in 0:D-1, j in 0:D-1
      i_idx = unravelIndex(i, dims)
      j_idx = unravelIndex(j, dims)
      for (k, i_) in enumerate(i_idx)
         dim = dims[k]
         idx_start = (cdims[k] - dim) * 2
         idx_real = idx_start + i_
         idx_imag = idx_start + dim + i_
         push!(indexmap[i+1,j+1], (idx_real, idx_imag, 1) )
      end
      for (k, j_) in enumerate(j_idx)
         dim = dims[k]
         idx_start = (cdims[k] - dim) * 2
         idx_real = idx_start + j_
         idx_imag = idx_start + dim + j_
         push!(indexmap[i+1,j+1], (idx_real, idx_imag, -1) )
      end
   end
   return indexmap
end

function projectDensityMat(X)
   # Ensure the matrix is Hermitian
   X = (X + X') / 2
   eigvals, eigvecs = eigen(X)
   maxndex = argmax(real(eigvals))   # Index of the maximum eigenvalue
   x = eigvecs[:, maxndex]     # Corresponding eigenvector
   # Construct xxᵀ
   principle = x * x'
   principle /= tr(principle)

   # Normalize the trace to 1
   return principle
end

function cartIndex(dims::Vector{Int64}, ind::Int64)
    tind = ind - 1
    carinds = fill(0, length(dims))
    size_of_dim = prod(dims)
    for (i,dim) in enumerate(dims)
       size_of_dim = div(size_of_dim, dim)  # Product of dimensions after the current one
       current_idx = div(tind, size_of_dim) + 1  # Calculate index for this dimension
       carinds[i] = current_idx
       tind %= size_of_dim  # Update remaining_idx to the modulus
    end
    return carinds
end


 # construct interpolation affine function
 function affine(var1, var2, bounds1var1, bounds1var2, bounds2var1, bounds2var2)
    var1_ = var1
    if bounds1var1 == bounds2var1
       var1_ = bounds1var1
    end
    var2_ = var2
    if bounds1var2 == bounds2var2
       var2_ = bounds1var2
    end
    return [var1_ * bounds1var2 + var2_ * bounds1var1 - bounds1var1 * bounds1var2, var1_ * bounds2var2 + var2_ * bounds2var1 - bounds2var1 * bounds2var2 ]
 end

"""
    subFactorBounds(fixvars, ZBs, subZinds, js, ks) -> bounds

Collect the `[:L, :U]` bounds of the two sub-factor entries that multiply to
form one entry of a parent `Z`, honouring any variable already fixed at this
node. Returned as `bounds[part, dir][i]` for `part in (:RE, :IM)`,
`dir in (:L, :U)` and `i in 1:2`.

Shared by the complex McCormick constraints and by the branching score, which
must convexify exactly the same bilinear terms.
"""
function subFactorBounds(fixvars, ZBs, subZinds, js, ks)
    bounds = Dict((:RE, :L) => zeros(2), (:RE, :U) => zeros(2),
                  (:IM, :L) => zeros(2), (:IM, :U) => zeros(2))
    for (i, (j, k)) in enumerate(zip(js, ks))
        subZind = subZinds[i]
        for part in (:RE, :IM)
            if haskey(fixvars, (subZind, part, j, k))
                fixed = fixvars[(subZind, part, j, k)]
                bounds[part, :L][i] = fixed
                bounds[part, :U][i] = fixed
            else
                bounds[part, :L][i] = ZBs[subZind][part, :L][j, k]
                bounds[part, :U][i] = ZBs[subZind][part, :U][j, k]
            end
        end
    end
    return bounds
end
