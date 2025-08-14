using Ket

T = Float64
N = 3
dims = Tuple(fill(2, N))
ρ = Ket.state_dicke(Complex{T}, 1, N)
