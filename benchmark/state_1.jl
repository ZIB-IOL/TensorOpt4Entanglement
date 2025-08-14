using Ket

name = "Dicke_4_1"
T = Float64
N = 4
dims = Tuple(fill(2, N))
ρ = Ket.state_dicke(Complex{T}, 1, N)