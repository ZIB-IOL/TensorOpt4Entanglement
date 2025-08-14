using Ket

name = "GHZ_5_2"
T = Float64
N = 5
dims = Tuple(fill(2, N))
ρ = Ket.state_ghz(Complex{T}, 2, N)