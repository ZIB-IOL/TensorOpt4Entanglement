using Ket
using LinearAlgebra

name = "Dicke_4_1"
T = Float64
N = 4
dims = Tuple(fill(2, N))
P1 = Matrix{Complex{T}}(Ket.state_dicke(Complex{T}, 1, N))


dimH = 2^N
P2 = Matrix{Complex{T}}(I, dimH, dimH) / dimH

print(tr(P1), tr(P2))


Mout = P1
Min = P2
Mdir = Mout - Min

nrank = 10
vs = [[randn(Complex{T}, dims[i]) for i in 1:N] for _ in 1:nrank]

nsubs = 4

function build_sT(vs, nrank, nsubs)
    sT = 0.0
    for i in 1:nrank
        xblock = vs[i]
        prod = 1
        for j in 1:nsubs
            x = xblock[j]
            xx = x * x'
            prod = kron(prod, xx)
        end
        if i == 1
            sT = prod
        else
            sT += prod
        end
    end
    return sT
end

sT = build_sT(vs,  nrank, nsubs)

print(tr(sT), "\n")
r = dot(real(Mout), real(sT)) + dot(imag(Mout), imag(sT))
if r + 1 <= 0
    sT = sT / abs(r)
end
print(dot(real(Mout), real(sT)) + dot(imag(Mout), imag(sT)), "\n")
print(dot(real(Mdir), real(sT)) + dot(imag(Mdir), imag(sT)), "\n")