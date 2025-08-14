using LinearAlgebra
using Zygote

function compute_inner_product_map(A, C, D)
    """
    Compute the map [i,j] -> ⟨A ⊗ E_ij ⊗ C, D⟩
    
    Args:
        A: m×m symmetric matrix
        C: p×p symmetric matrix  
        D: (m*n*p)×(m*n*p) matrix where n is the dimension we're mapping over
    
    Returns:
        result: n×n matrix where result[i,j] = ⟨A ⊗ E_ij ⊗ C, D⟩
    """
    m, _ = size(A)
    p, _ = size(C)
    total_size = size(D, 1)
    n = total_size ÷ (m * p)  # Infer n from dimensions
    
    # Pre-allocate result matrix
    result = zeros(n, n)
    
    # Compute for each (i,j) pair
    for i in 1:n, j in 1:n
        # Create E_ij: n×n matrix with 1 at position (i,j), 0 elsewhere
        E_ij = zeros(n, n)
        E_ij[i, j] = 1.0
        
        # Compute A ⊗ E_ij ⊗ C
        kron_product = kron(A, kron(E_ij, C))
        
        # Compute inner product (Frobenius inner product)
        result[i, j] = tr(kron_product' * D)
    end
    
    return result
end

function compute_inner_product_map_efficient(A, C, D)
    """
    More efficient version using the block structure of Kronecker products.
    
    For A ⊗ E_ij ⊗ C, the element at position ((α-1)*n*p + (i-1)*p + β, (γ-1)*n*p + (j-1)*p + δ)
    is A[α,γ] * E_ij[i,j] * C[β,δ] = A[α,γ] * δ_ii * δ_jj * C[β,δ]
    where δ_ii = 1 if row corresponds to i, 0 otherwise
    """
    m, _ = size(A)
    p, _ = size(C)
    total_size = size(D, 1)
    n = total_size ÷ (m * p)
    
    result = zeros(n, n)
    
    for i in 1:n, j in 1:n
        inner_sum = 0.0
        for α in 1:m, β in 1:p, γ in 1:m, δ in 1:p
            # The Kronecker product A ⊗ E_ij ⊗ C has structure:
            # Row index: (α-1)*n*p + (i-1)*p + β
            # Col index: (γ-1)*n*p + (j-1)*p + δ
            row_idx = (α-1)*n*p + (i-1)*p + β
            col_idx = (γ-1)*n*p + (j-1)*p + δ
            inner_sum += A[α, γ] * C[β, δ] * D[row_idx, col_idx]
        end
        result[i, j] = inner_sum
    end
    
    return result
end


function partialfunc(X, kron_prod_left, kron_prod_right, H)
    kron_prod = kron(kron_prod_left, X , kron_prod_right)
    return dot(real(H), real(kron_prod)) + dot(imag(H), imag(kron_prod))
end


function compute_inner_product_map_vectorized(A, C, D)
    """
    Most efficient vectorized version using Einstein summation pattern.
    
    We use the fact that:
    ⟨A ⊗ E_ij ⊗ C, D⟩ = Σ_{α,β,γ,δ} A[α,γ] * C[β,δ] * D[row,col]
    where row = (α-1)*n*p + (i-1)*p + β + 1
          col = (γ-1)*n*p + (j-1)*p + δ + 1
    """
    m, _ = size(A)
    p, _ = size(C)
    total_size = size(D, 1)
    n = total_size ÷ (m * p)
    
    result = zeros(n, n)
    
    # Pre-compute all possible row and column indices
    for i in 1:n, j in 1:n
        # Use broadcasting to compute all combinations at once
        α_range = 1:m
        β_range = 1:p  
        γ_range = 1:m
        δ_range = 1:p
        
        # Compute row and column indices for all combinations
        rows = [(α-1)*n*p + (i-1)*p + β for α in α_range, β in β_range]
        cols = [(γ-1)*n*p + (j-1)*p + δ for γ in γ_range, δ in δ_range]
        
        # Extract the relevant submatrix from D
        D_sub = D[rows[:], cols[:]]  # m*p × m*p matrix
        
        # Reshape to separate the A and C components
        D_reshaped = reshape(D_sub, m, p, m, p)
        
        # Contract with A and C using Einstein summation
        # result[i,j] = Σ_{α,β,γ,δ} A[α,γ] * C[β,δ] * D[α,β,γ,δ]
        result[i, j] = sum(A[α,γ] * C[β,δ] * D_reshaped[α,β,γ,δ] 
                          for α in 1:m, β in 1:p, γ in 1:m, δ in 1:p)
    end
    
    return result
end

# Example usage
function example_usage()
    # Define dimensions
    m, n, p = 3, 4, 2
    
    # Create symmetric matrices A and C
    A = rand(m, m)
    A = (A + A') / 2  # Make symmetric
    
    C = rand(p, p) 
    C = (C + C') / 2  # Make symmetric
    
    # Create a random matrix D of appropriate size
    D = rand(m*n*p, m*n*p)
    
    println("Matrix dimensions:")
    println("A: $(size(A)), C: $(size(C)), D: $(size(D))")
    
    # Compute using different methods
    result1 = compute_inner_product_map(A, C, D)
    result2 = compute_inner_product_map_efficient(A, C, D)
    result3 = compute_inner_product_map_vectorized(A, C, D)
    result4 = gradient(p -> partialfunc(p, A, C, D), rand(n,n))[1]
   
    println(result1)
    println(result2)
    z = randn(4,4)
    println(sum(result3 .* z))
    println(sum(result4 .* z))
    println(sum(kron(A, z, C) .* D))
    
    println("\nResults match: ", 
            isapprox(result1, result2, rtol=1e-10) && 
            isapprox(result2, result3, rtol=1e-10))
    
    return result1, A, C, D
end

# Run example
result, A, C, D = example_usage()
