"""
White-noise entanglement bounds using primal-dual geometric reconstruction.
Adapted from https://github.com/ZIB-IOL/EntanglementDetection.jl.
This source folder contains the required numerical core and can be included
directly; it does not load ExactEntanglement or an external source checkout.
"""
module PDGR

include("EntanglementDetection/EntanglementDetection.jl")
include("white_noise_bounds.jl")

using .CertifiedWhiteNoiseBounds: NoiseBoundConfig, white_noise_bounds,
    white_noise_sep_bound, white_noise_ent_bound, print_bound_summary

const Options = NoiseBoundConfig

"""
    solve(rho, dims; options=Options(), structure=:full, mode=:both, kwargs...)

Bound the threshold for `(1-p)rho + p*I/prod(dims)`. `rho` may be a density
matrix or a ket. Keyword options override fields of `options`. The result
contains `ent_bound`, `sep_bound`, `gap`, `status`, certificates and history.

`time_limit` is cooperative: completed certificates survive a timeout;
an incomplete witness enumeration is discarded. Compilation, allocation and
individual linear-algebra calls can exceed the requested wall-clock limit.
"""
function solve(rho, dims; options::Options=Options(), structure=:full,
               mode::Symbol=:both, kwargs...)
    all(d -> d isa Integer, dims) || throw(ArgumentError("dims must contain integers"))
    settings = (; (k => getfield(options, k) for k in fieldnames(Options))...)
    config = Options(; merge(settings, (; kwargs...))...)
    return white_noise_bounds(rho, Tuple(Int.(dims)); structure, mode, config)
end

export Options, solve, white_noise_bounds, white_noise_sep_bound,
       white_noise_ent_bound, print_bound_summary
end
