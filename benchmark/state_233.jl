using Ket

name = "Dicke_3_2"
T = Float64
N = 3
dims = Tuple(fill(2, N))
ρ = Ket.state_dicke(Complex{T}, 2, N)
