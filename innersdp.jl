using Ket
using LinearAlgebra
using MosekTools
using JuMP

T = Float64
N = 1
dims = [2]

function random_non_psd_matrix(dim::Int)
    A = randn(Complex{T}, dim, dim)
    return A + A'  # This is Hermitian but not guaranteed to be PSD
end


dimH = 2^N
P1 = random_non_psd_matrix(dimH)
P1 = P1 / tr(P1)  # Normalize to make it a valid density matrix
P2 = Matrix{Complex{T}}(I, dimH, dimH) / dimH

print(tr(P1), tr(P2))


Mout = P1
Minc = P2
Mdir = Mout - Minc

nrank = 10
vs = [[randn(Complex{T}, dims[i]) for i in 1:N] for _ in 1:nrank]

nsubs = 4

function build_sT(vs, nrank, nsubs)
    sT = 0.0
    for i in 1:nrank
        xblock = vs[i]
        prod = 1
        x = xblock[1]
        xx = x * x'
        prod = kron(prod, xx)
        if i == 1
            sT = prod
        else
            sT += prod
        end
    end
    return sT
end

function primal(Mout, Mdir, dim)
    # Compute the primal objective value
    println("Computing primal objective value...")
    model = Model(Mosek.Optimizer)
    set_silent(model)
    @variable(model, t >= 0)
    XR = @variable(model, [1:dim, 1:dim], Symmetric)
    XI = @variable(model, [1:dim, 1:dim] in SkewSymmetricMatrixSpace())
    @constraint(model, [XR XI; -XI XR] in PSDCone())
    @objective(model, Min, t)
    @constraint(model, -Mdir *t  +  Mout == XR + im * XI)
    optimize!(model)
    print(solution_summary(model))
    println("\n\n\n\n\n")
    return objective_value(model)
end

function primal2(Mout, Mdir, dim)
    # Compute the primal objective value
    println("Computing primal objective value with t = 1...")
    model = Model(Mosek.Optimizer)
    set_silent(model)
    @variable(model, t >= 0)
    XR = @variable(model, [1:dim, 1:dim], Symmetric)
    XI = @variable(model, [1:dim, 1:dim] in SkewSymmetricMatrixSpace())
    @constraint(model, [XR XI; -XI XR] in PSDCone())
    @objective(model, Min, t)
    @constraint(model, -Mdir *t  +  Mout == XR + im * XI)
    @constraint(model, t == 1)
    optimize!(model)
    print(solution_summary(model))
    println("\n\n\n\n\n")
    return objective_value(model)
end
print("primalfix:", primal2(Mout, Mdir, dims[1]), "\n")

print("primal:", primal(Mout, Mdir, dims[1]), "\n")

function dual(Mout, Mdir, dim)
    # Compute the primal objective value
    println("Computing dual objective value...")
    model = Model(Mosek.Optimizer)
    set_silent(model)
    XR = @variable(model, [1:dim, 1:dim], Symmetric)
    XI = @variable(model, [1:dim, 1:dim] in SkewSymmetricMatrixSpace())
    @constraint(model, [XR XI; -XI XR] in PSDCone())
    @objective(model, Max, dot(real(Mout), -XR) + dot(imag(Mout), -XI))
    @constraint(model, 1 - dot(real(Mdir), -XR) - dot(imag(Mdir), -XI) >= 0)
    optimize!(model)
    print(solution_summary(model))
    println("\n\n\n\n\n")
    return objective_value(model)
end

print("dual:", dual(Mout, Mdir, dims[1]), "\n")

function dualfix(Minc, Mout, Mdir, dim, sT)
    # Compute the primal objective value
    println("Computing dual objective value with fixed sT...")
    model = Model(Mosek.Optimizer)
    set_silent(model)
    XR = @variable(model, [1:dim, 1:dim], Symmetric)
    XI = @variable(model, [1:dim, 1:dim] in SkewSymmetricMatrixSpace())
    #@constraint(model, [XR XI; -XI XR] in PSDCone())
    @constraint(model, XR == real(sT))
    @constraint(model, XI == imag(sT))
    @objective(model, Max, dot(real(Mout), XR) + dot(imag(Mout), XI))
    @constraint(model, dot(real(Mdir), XR) + dot(imag(Mdir), XI) <= 1)
    optimize!(model)
    print(solution_summary(model))
        println("\n\n\n\n\n")
    return objective_value(model)
end

sT = build_sT(vs,  nrank, nsubs)

print(tr(sT), "\n")
r = dot(real(Mdir), real(sT)) + dot(imag(Mdir), imag(sT))

#  1 + r >= 0, r >= -1
if r <= -1
    sT = sT / abs(r)
end

print(dualfix(Minc, Mout, Mdir, dims[1], sT), "\n")
print( - dot(real(Mout), real(sT)) - dot(imag(Mout), imag(sT)), "\n")
print(1 + dot(real(Mdir), real(sT)) + dot(imag(Mdir), imag(sT)), "\n")
