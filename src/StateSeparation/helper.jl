function linearIndex(dims::Vector{Int64}, inds::Vector{Int64})
    @assert length(dims) = length(inds)
    tind = 1
    presize = 1
    for (dim, ind) in reverse(zip(dims, inds))
       tind += (ind - 1) * presize
       presize *= dim
    end
    return tind
end

function antiPart(part)
   return part == :RE ? (:IM, -1) : (:RE, 1)
end
# Projection onto the manifold of Hermitian positive semi-definite matrices with trace 1
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

 safediv(x, y, tol) = ( x / (y > 0 ? max(y, tol) : min(y, -tol)))

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

 function replace_element!(arr, i, j, k)
   idx = findfirst(==(i), arr)  # Find index of i
   if isnothing(idx)
       return arr  # If i is not found, return the original array
   end
   splice!(arr, idx, [j, k]) # Replace i with j, k
   return arr
end