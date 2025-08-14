using Ket

T = Float64
N = 4
dims = Tuple(fill(2, N))
ρ = Ket.state_ghz(Complex{T}, 2, N)