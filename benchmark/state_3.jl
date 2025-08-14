using Ket

name = ""
T = Float64
N = 4
dims = Tuple(fill(2, N))
phi = Complex{T}[
    1, 1, 1, -1,
    1, 1, -1, 1,
    1, 1, 1, -1,
   -1, -1, 1, -1
]
ρ = phi * phi' / norm(phi)^2