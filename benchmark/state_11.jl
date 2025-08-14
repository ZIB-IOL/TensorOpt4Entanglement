using Ket

T = Float64
N = 5
dims = Tuple(fill(2, N))
ρ = Ket.state_dicke(Complex{T}, 1, N)