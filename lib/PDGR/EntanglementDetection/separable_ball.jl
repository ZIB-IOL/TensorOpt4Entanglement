function separable_ball_radius(::Type{T}, dims::NTuple{N, Int}) where {T <: Number, N}
    N == 1 && throw(ArgumentError("At least two subsystems are required."))
    if length(unique(dims)) ≤ 1 # symmetry
        if dims[1] == 2 && N ≥ 3 # N-qubits (https://arxiv.org/abs/quant-ph/0601201)
            return sqrt(54 / 17) * 6^(-N / 2)
        else # N-qudits (https://arxiv.org/abs/quant-ph/0409095)
            return 1 / sqrt((2 * dims[1] - 1)^(N - 2) * (dims[1]^2 - 1) * dims[1]^N + dims[1]^N)
        end
    else # asymmetry (https://doi.org/10.1103/PhysRevA.68.042312)
        return 1 / (2^(N / 2 - 1) * prod(dims))
    end
end
separable_ball_radius(dims) = separable_ball_radius(Float64, dims)
export separable_ball_radius

function separable_ball_radius(::Type{T}, lmo::KSeparableLMO{T, N}) where {T <: Number, N}
    return maximum([separable_ball_radius(T, lmo.lmos[i].dims) for i in eachindex(lmo.lmos)])
end

function separable_ball_radius(::Type{T}, lmo::AlternatingSeparableLMO{T, N}) where {T <: Number, N}
    return separable_ball_radius(T, lmo.dims)
end
