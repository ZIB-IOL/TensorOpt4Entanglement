module CertifiedWhiteNoiseBounds

using ..EntanglementDetection
using FrankWolfe
using LinearAlgebra
using Random
using Printf
using DoubleFloats

export NoiseBoundConfig, white_noise_bounds, white_noise_sep_bound, white_noise_ent_bound, print_bound_summary

"""
Numerical settings for `white_noise_bounds`.

Defaults match results/Liding/both.jl in the source repository.
Every numerical setting can be changed by the caller.
"""
Base.@kwdef struct NoiseBoundConfig
    # arithmetic
    T::DataType = Float64
    state_atol::Float64 = 1e-10

    # requested accuracy of the final noise interval [p_ent, p_sep]
    bound_atol::Float64 = 1e-3
    max_steps::Int = 15

    # Frank-Wolfe / CG approximation of the separable set
    fw_epsilon::Float64 = 1e-7
    fw_max_iteration::Int = 10^6
    fw_callback_iter::Int = 10^7
    fw_lazy::Bool = true
    fw_shortcut::Bool = false
    fw_shortcut_scale::Float64 = 10.0
    fw_warm_start::Bool = true

    # alternating LMO used inside CG
    lmo_nb::Int = 10
    lmo_max_iter::Int = 10^3
    lmo_threshold::Union{Nothing, Float64} = nothing
    lmo_parallelism::Bool = false

    # rigorous epsilon-net witness
    witness_max_length::Int = 10^7
    witness_min_eta::Float64 = 0.0
    witness_tol::Float64 = 0.0
    witness_margin::Float64 = 1e-12

    # Cooperative wall-clock budget; Inf preserves the upstream execution.
    time_limit::Float64 = Inf

    # deterministic/reproducible run
    seed::Int = 0
    verbose::Int = 1
end

# Backward-compatible qualified alias. Not exported to avoid name collisions.
const BoundConfig = NoiseBoundConfig

# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

function _resolve_k(structure, N::Int)
    if structure === :full || structure === :fully_separable
        return N
    elseif structure === :gme || structure === :biseparable
        return 2
    elseif structure isa Integer
        k = Int(structure)
        2 <= k <= N || throw(ArgumentError("structure = k must satisfy 2 <= k <= N=$N"))
        return k
    else
        throw(ArgumentError(
            "structure must be :full, :gme/:biseparable, or an integer k in 2:N"
        ))
    end
end

function _structure_name(k::Int, N::Int)
    if k == N
        return "full separability"
    elseif k == 2
        return "2-separability (GME boundary)"
    else
        return "$(k)-separability"
    end
end

function _prepare_state(ρ0::AbstractMatrix, dims::NTuple{N, Int}, cfg::NoiseBoundConfig) where {N}
    D = prod(dims)
    size(ρ0) == (D, D) || throw(ArgumentError(
        "state size $(size(ρ0)) is incompatible with dims=$dims (total dimension $D)"
    ))

    ρ64 = Matrix{ComplexF64}(ρ0)
    norm(ρ64 - ρ64') <= cfg.state_atol || throw(ArgumentError(
        "input state is not Hermitian within state_atol=$(cfg.state_atol)"
    ))

    trρ = real(tr(ρ64))
    abs(trρ - 1.0) <= cfg.state_atol || throw(ArgumentError(
        "input state must have trace 1; got Tr(rho)=$trρ"
    ))

    λmin = eigmin(Hermitian((ρ64 + ρ64') / 2))
    λmin >= -cfg.state_atol || throw(ArgumentError(
        "input state is not positive semidefinite within state_atol=$(cfg.state_atol); lambda_min=$λmin"
    ))

    T = cfg.T
    ρ = Matrix{Complex{T}}(ρ0)
    # Remove harmless non-Hermitian roundoff without renormalizing the user's state.
    ρ = (ρ + ρ') / T(2)
    return ρ
end

function _white_noise(::Type{T}, D::Int) where {T <: Real}
    return Matrix{Complex{T}}(I, D, D) / T(D)
end

_mix(ρ0, ω, p) = (one(p) - p) * ρ0 + p * ω

# -----------------------------------------------------------------------------
# Separable-set LMOs
# -----------------------------------------------------------------------------

function _make_search_lmo(::Type{T}, dims::NTuple{N, Int}, k::Int, cfg::NoiseBoundConfig) where {T <: Real, N}
    threshold = isnothing(cfg.lmo_threshold) ? Base.rtoldefault(T) : T(cfg.lmo_threshold)
    kwargs = (
        nb = cfg.lmo_nb,
        threshold = threshold,
        max_iter = cfg.lmo_max_iter,
        parallelism = cfg.lmo_parallelism,
    )

    if k == N
        return EntanglementDetection.AlternatingSeparableLMO(T, dims; kwargs...)
    else
        return EntanglementDetection.KSeparableLMO(
            T,
            dims,
            k;
            LMO = EntanglementDetection.AlternatingSeparableLMO,
            kwargs...,
        )
    end
end

function _make_rigorous_lmo(::Type{T}, dims::NTuple{N, Int}, k::Int, cfg::NoiseBoundConfig) where {T <: Real, N}
    kwargs = (
        max_length = cfg.witness_max_length,
        min_η = T(cfg.witness_min_eta),
    )

    if k == N
        return EntanglementDetection.EnumeratingSeparableLMO(T, dims; kwargs...)
    else
        # Each k-partition gets its own rigorous EnumeratingSeparableLMO.
        return EntanglementDetection.KSeparableLMO(
            T,
            dims,
            k;
            LMO = EntanglementDetection.EnumeratingSeparableLMO,
            kwargs...,
        )
    end
end

# -----------------------------------------------------------------------------
# One CG solve at a fixed white-noise level
# -----------------------------------------------------------------------------

"""
Normalize a warm-start active set returned by different FrankWolfe versions.

Recent FrankWolfe versions return an `AbstractActiveSet`, whereas some versions
used together with EntanglementDetection return a plain `Vector{Any}` of
`(weight, atom)` tuples.  Passing that vector back into BPCG is interpreted as a
single iterate and fails because its element type is `Any`.

This helper converts the legacy vector representation into the quadratic
product-caching active set expected by `separable_distance`.  If the returned
object cannot be interpreted safely, warm start is dropped rather than risking
an invalid solver state.
"""
function _normalize_active_set(active_set, ρp, lmo, cfg::NoiseBoundConfig)
    cfg.fw_warm_start || return nothing
    active_set === nothing && return nothing

    if isdefined(FrankWolfe, :AbstractActiveSet) && isa(active_set, getfield(FrankWolfe, :AbstractActiveSet))
        return active_set
    end

    if active_set isa AbstractVector
        isempty(active_set) && return nothing

        function unpack(entry)
            if entry isa Pair
                return entry.first, entry.second
            elseif entry isa Tuple && length(entry) == 2
                return entry[1], entry[2]
            else
                return nothing
            end
        end

        first_pair = unpack(first(active_set))
        if first_pair === nothing
            cfg.verbose >= 1 && @warn "Dropping incompatible warm-start active set" type=typeof(active_set)
            return nothing
        end

        w0, a0 = first_pair
        R = cfg.T
        AT = typeof(a0)
        tuples = Vector{Tuple{R,AT}}(undef, length(active_set))

        for i in eachindex(active_set)
            item = unpack(active_set[i])
            if item === nothing
                cfg.verbose >= 1 && @warn "Dropping malformed warm-start active set"
                return nothing
            end
            wi, ai = item
            if !(ai isa AT)
                cfg.verbose >= 1 && @warn "Dropping heterogeneous warm-start active set" expected=AT got=typeof(ai)
                return nothing
            end
            tuples[i] = (R(wi), copy(ai))
        end

        C = correlation_tensor(ρp, lmo.dims, lmo.matrix_basis)
        try
            return FrankWolfe.ActiveSetQuadraticProductCaching(tuples, I, -C)
        catch err
            cfg.verbose >= 1 && @warn "Could not rebuild warm-start active set; continuing without warm start" exception=(err, catch_backtrace())
            return nothing
        end
    end

    cfg.verbose >= 1 && @warn "Dropping unsupported warm-start active set" type=typeof(active_set)
    return nothing
end

function _distance_at_p(ρp, dims, lmo, active_set, cfg::NoiseBoundConfig)
    Random.seed!(cfg.seed)
    return separable_distance(
        ρp,
        dims,
        lmo;
        active_set = active_set,
        fw_algorithm = FrankWolfe.blended_pairwise_conditional_gradient,
        max_iteration = cfg.fw_max_iteration,
        callback_iter = cfg.fw_callback_iter,
        epsilon = cfg.fw_epsilon,
        lazy = cfg.fw_lazy,
        shortcut = cfg.fw_shortcut,
        shortcut_scale = cfg.fw_shortcut_scale,
        noise_mixture = false,
        trajectory = false,
        verbose = cfg.verbose >= 2 ? 2 : 0,
    )
end

# -----------------------------------------------------------------------------
# Rigorous witness, including k-separable structures
# -----------------------------------------------------------------------------

function _single_partition_support(∇, dir, rigorous_lmo)
    x = FrankWolfe.compute_extreme_point(rigorous_lmo, dir)
    ϕ = density_matrix(x, rigorous_lmo.dims, rigorous_lmo.matrix_basis)
    α = real(dot(ϕ, ∇))
    η = EntanglementDetection.radius_inner(rigorous_lmo)
    return (α = α, η = η, ϕ = ϕ)
end

function _k_partition_support(∇, dir, rigorous_lmo::EntanglementDetection.KSeparableLMO)
    # For S_k = conv(union_i S_i), a witness must be valid for every k-partition.
    # We therefore correct the support function independently for each partition
    # and take the smallest rigorous lower support bound.
    distance = norm(∇)
    best = nothing

    for i in eachindex(rigorous_lmo.lmos)
        child = rigorous_lmo.lmos[i]
        partition = rigorous_lmo.partitions[i]

        grouped_dir = EntanglementDetection.group_dims(dir, child.dims, partition)
        x_grouped = FrankWolfe.compute_extreme_point(child, grouped_dir)
        x = EntanglementDetection.ungroup_dims(x_grouped, rigorous_lmo.dims, partition)
        ϕ = density_matrix(x, rigorous_lmo.dims, rigorous_lmo.matrix_basis)

        α = real(dot(ϕ, ∇))
        η = EntanglementDetection.radius_inner(child)
        ε = (one(η) - η) * distance
        support_lower = α - ε

        if best === nothing || support_lower < best.support_lower
            best = (
                α = α,
                η = η,
                ε = ε,
                support_lower = support_lower,
                ϕ = ϕ,
                partition = partition,
            )
        end
    end

    return best
end

function _rigorous_witness(ρ, σ, rigorous_lmo, cfg::NoiseBoundConfig)
    ∇ = σ - ρ
    distance = norm(∇)

    if !isfinite(Float64(distance)) || distance <= eps(Float64)
        return (
            valid = false,
            detects_probe = false,
            certified = false,
            W = nothing,
            σ = σ,
            ϕ = nothing,
            distance = distance,
            η = NaN,
            val_rho = NaN,
            val_sigma = NaN,
            support_lower = NaN,
            partition = nothing,
            reason = "zero_or_nonfinite_distance",
        )
    end

    dir = correlation_tensor(∇, rigorous_lmo.dims, rigorous_lmo.matrix_basis)

    support = if rigorous_lmo isa EntanglementDetection.KSeparableLMO
        _k_partition_support(∇, dir, rigorous_lmo)
    else
        s = _single_partition_support(∇, dir, rigorous_lmo)
        ε = (one(s.η) - s.η) * distance
        (
            α = s.α,
            η = s.η,
            ε = ε,
            support_lower = s.α - ε,
            ϕ = s.ϕ,
            partition = nothing,
        )
    end

    D = size(ρ, 1)
    Id = Matrix{eltype(ρ)}(I, D, D)
    W = ∇ - support.support_lower * Id
    W ./= distance

    val_rho = Float64(real(dot(W, ρ)))
    val_sigma = Float64(real(dot(W, σ)))
    tol = cfg.witness_tol

    # The epsilon-net correction establishes validity on the whole requested
    # separable set.  val_sigma is an additional numerical sanity check.
    valid = (val_sigma >= -tol)
    detects_probe = (val_rho < -tol)
    certified = valid && detects_probe

    return (
        valid = valid,
        detects_probe = detects_probe,
        certified = certified,
        W = W,
        σ = σ,
        ϕ = support.ϕ,
        distance = Float64(distance),
        η = Float64(support.η),
        val_rho = val_rho,
        val_sigma = val_sigma,
        support_lower = Float64(support.support_lower / distance),
        partition = support.partition,
        reason = certified ? "certified" : (!valid ? "validity_sanity_check_failed" : "probe_not_detected"),
    )
end

# -----------------------------------------------------------------------------
# Rigorous lower/upper updates from the same CG result
# -----------------------------------------------------------------------------

function _fixed_witness_projection(ρ0, ω, witness, cfg::NoiseBoundConfig)
    witness.W === nothing && return (usable = false, p = NaN, p_cross = NaN, reason = "no_witness")
    witness.valid || return (usable = false, p = NaN, p_cross = NaN, reason = "invalid_witness")

    W = witness.W
    f0 = Float64(real(dot(W, ρ0)))
    f1 = Float64(real(dot(W, ω)))
    slope = f1 - f0

    if !(isfinite(f0) && isfinite(f1) && isfinite(slope))
        return (usable = false, p = NaN, p_cross = NaN, reason = "nonfinite")
    end
    if f0 >= -cfg.witness_tol
        return (usable = false, p = NaN, p_cross = NaN, reason = "rho0_not_detected")
    end
    if slope <= 0
        return (usable = false, p = NaN, p_cross = Inf, reason = "nonpositive_slope")
    end

    p_cross = -f0 / slope
    p_cert = (-cfg.witness_margin - f0) / slope

    if p_cert < 0
        return (usable = false, p = NaN, p_cross = p_cross, reason = "negative_crossing")
    end
    if p_cert >= 1.0
        return (usable = false, p = NaN, p_cross = p_cross, reason = "crossing_at_or_above_one")
    end

    ρtest = _mix(ρ0, ω, cfg.T(p_cert))
    val = Float64(real(dot(W, ρtest)))
    usable = val < -cfg.witness_tol

    return (
        usable = usable,
        p = p_cert,
        p_cross = p_cross,
        value = val,
        f0 = f0,
        f1 = f1,
        reason = usable ? "ok" : "candidate_not_strictly_negative",
    )
end

function _geometric_sep_bound(p::Float64, res, lmo, cfg::NoiseBoundConfig)
    # Matrix-input separable_distance returns primal = ||rho - sigma||_HS^2.
    δ = sqrt(max(Float64(res.primal), 0.0))
    r = Float64(EntanglementDetection.separable_ball_radius(cfg.T, lmo))

    if !(isfinite(δ) && isfinite(r) && r > 0)
        return (usable = false, p = 1.0, distance = δ, radius = r)
    end

    ε = δ / r
    p_sep = (p + ε) / (1.0 + ε)
    p_sep = clamp(p_sep, p, 1.0)

    return (
        usable = true,
        p = p_sep,
        distance = δ,
        radius = r,
        epsilon = ε,
    )
end

# -----------------------------------------------------------------------------
# Public driver
# -----------------------------------------------------------------------------

"""
    white_noise_bounds(ρ0, dims; structure=:full, mode=:both,
                       config=NoiseBoundConfig())

Compute white-noise robustness bounds for

    ρ(p) = (1-p)ρ0 + p I/D,     0 <= p <= 1.

`mode` controls which certificate is computed:
- `:both` : rigorous entanglement lower bound + rigorous separability upper bound.
- `:sep`  : separability upper bound only; the expensive rigorous witness is skipped.
- `:ent`  : entanglement lower bound only; geometric reconstruction is skipped.

For `mode=:sep`, `sep_bound` is rigorous, while `search_low` is only an
algorithmic search anchor used to locate a tighter upper bound.  It is NOT an
entanglement certificate.  Consequently no certified two-sided gap is reported
in sep-only mode.
"""
function white_noise_bounds(ρ0_in::AbstractMatrix, dims::NTuple{N, Int};
                            structure=:full, mode::Symbol=:both,
                            config::NoiseBoundConfig=NoiseBoundConfig()) where {N}
    validate_config(config)
    deadline = time_ns() / 1e9 + config.time_limit
    return task_local_storage(:PDGR_deadline, deadline) do
        _white_noise_bounds(ρ0_in, dims; structure, mode, config)
    end
end

function validate_config(cfg::NoiseBoundConfig)
    (cfg.time_limit >= 0 && !isnan(cfg.time_limit)) ||
        throw(ArgumentError("time_limit must be nonnegative or Inf"))
    cfg.T <: AbstractFloat || throw(ArgumentError("T must be a floating-point type"))
    for field in (:state_atol, :bound_atol, :fw_epsilon)
        value = getfield(cfg, field)
        (isfinite(value) && value > 0) || throw(ArgumentError("$field must be positive and finite"))
    end
    for field in (:max_steps, :fw_max_iteration, :fw_callback_iter, :lmo_nb,
                  :lmo_max_iter, :witness_max_length)
        getfield(cfg, field) >= 1 || throw(ArgumentError("$field must be positive"))
    end
    for field in (:witness_tol, :witness_margin)
        value = getfield(cfg, field)
        (isfinite(value) && value >= 0) || throw(ArgumentError("$field must be finite and nonnegative"))
    end
    0 <= cfg.witness_min_eta <= 1 || throw(ArgumentError("witness_min_eta must be in [0,1]"))
    cfg.verbose >= 0 || throw(ArgumentError("verbose must be nonnegative"))
    return cfg
end

function _white_noise_bounds(
    ρ0_in::AbstractMatrix,
    dims::NTuple{N, Int};
    structure = :full,
    mode::Symbol = :both,
    config::NoiseBoundConfig = NoiseBoundConfig(),
) where {N}
    N >= 2 || throw(ArgumentError("at least two subsystems are required"))
    all(d -> d >= 2, dims) || throw(ArgumentError("all local dimensions must be >= 2"))
    config.bound_atol > 0 || throw(ArgumentError("bound_atol must be positive"))
    config.max_steps >= 1 || throw(ArgumentError("max_steps must be >= 1"))
    mode in (:both, :sep, :ent) || throw(ArgumentError("mode must be :both, :sep, or :ent"))

    do_sep = mode != :ent
    do_ent = mode != :sep

    k = _resolve_k(structure, N)
    T = config.T
    ρ0 = _prepare_state(ρ0_in, dims, config)
    D = prod(dims)
    ω = _white_noise(T, D)

    search_lmo = _make_search_lmo(T, dims, k, config)
    # This is usually the expensive object.  Do not even construct it in sep-only mode.
    rigorous_lmo = do_ent ? _make_rigorous_lmo(T, dims, k, config) : nothing

    # Rigorous certificates.
    lower = 0.0                  # used only when do_ent=true
    upper = 1.0                  # used only when do_sep=true; p=1 is rigorously separable

    # Search-only anchors.  These help locate the boundary when only one side is
    # requested.  They are never advertised as physical certificates.
    search_low = 0.0
    search_high = 1.0

    active_set = nothing
    best_witness = nothing
    best_sep = do_sep ? (p = 1.0, source = "maximally_mixed_state") : nothing
    history = NamedTuple[]

    probe = 0.0
    last_probe = NaN
    termination_status = nothing

    # In :both mode we first optimize the certified entanglement lower bound by
    # probing ONLY at `lower`.  If that certified lower stalls while the target
    # gap is still open, we freeze it and switch to an upper-only refinement
    # phase.  That second phase never changes `lower` and skips the expensive
    # rigorous witness entirely.
    upper_refinement = false
    upper_refine_low = 0.0
    upper_refine_high = 1.0

    if config.verbose >= 1
        println("============================================================")
        println(mode == :both ? "Certified white-noise bounds" :
                mode == :sep  ? "Certified white-noise separability bound" :
                                "Certified white-noise entanglement bound")
        println("dims      = ", dims)
        println("structure = ", _structure_name(k, N))
        println("mode      = ", mode)
        println("p range   = [0, 1]")
        if mode == :both
            println("target certified gap <= ", config.bound_atol)
        else
            println("target search resolution <= ", config.bound_atol)
        end
        println("============================================================")
    end

    try
        for step in 1:config.max_steps
            EntanglementDetection.check_time_limit()
            # Probe policy:
            #   :both / :ent main phase -> probe ONLY at the rigorously certified
            #                              lower bound.
            #   :both upper-refinement  -> lower is frozen; probe numerically inside
            #                              an upper-refinement bracket.
            #   :sep                    -> use the numerical search bracket.
            if mode == :sep
                probe = clamp(probe, search_low, search_high)
                if probe >= 1.0
                    probe = (search_low + search_high) / 2
                end

                min_move = max(config.bound_atol / 100, 1e-12)
                if isfinite(last_probe) && abs(probe - last_probe) <= min_move
                    probe = (search_low + search_high) / 2
                end
            elseif mode == :both && upper_refinement
                probe = clamp(probe, upper_refine_low, upper_refine_high)
                min_move = max(config.bound_atol / 100, 1e-12)
                if isfinite(last_probe) && abs(probe - last_probe) <= min_move
                    probe = (upper_refine_low + upper_refine_high) / 2
                end
            else
                probe = lower
            end
            last_probe = probe

            if config.verbose >= 1
                if mode == :both
                    phase_label = upper_refinement ? "upper refinement" : "certified-lower refinement"
                    @printf("\n[%d/%d] probe p = %.12g   current interval = [%.12g, %.12g]   phase = %s\n",
                        step, config.max_steps, probe, lower, upper, phase_label)
                elseif mode == :sep
                    @printf("\n[%d/%d] probe p = %.12g   search bracket = [%.12g, %.12g]\n",
                        step, config.max_steps, probe, search_low, search_high)
                else
                    @printf("\n[%d/%d] probe p = %.12g   search bracket = [%.12g, %.12g]\n",
                        step, config.max_steps, probe, lower, search_high)
                end
            end

            ρp = _mix(ρ0, ω, T(probe))
            res = _distance_at_p(ρp, dims, search_lmo, active_set, config)
            active_set = _normalize_active_set(res.active_set, ρp, search_lmo, config)
            cg_distance = sqrt(max(Float64(res.primal), 0.0))

            old_lower = lower
            old_upper = upper

            # ----- rigorous separability upper bound -----
            geo = do_sep ? _geometric_sep_bound(probe, res, search_lmo, config) : nothing
            if do_sep && geo.usable && geo.p < upper
                upper = geo.p
                best_sep = (
                    p = upper,
                    probe = probe,
                    distance = geo.distance,
                    radius = geo.radius,
                    epsilon = geo.epsilon,
                    σ = res.σ,
                    source = "geometric_reconstruction",
                )
            end

            # ----- rigorous entanglement lower bound -----
            # During the upper-only refinement phase in :both mode, the certified
            # lower bound is intentionally frozen and no witness is recomputed.
            ent_this_step = do_ent && !(mode == :both && upper_refinement)
            witness = nothing
            proj = nothing
            if ent_this_step
                witness = _rigorous_witness(ρp, res.σ, rigorous_lmo, config)
                proj = _fixed_witness_projection(ρ0, ω, witness, config)

                if witness.certified && probe >= lower
                    lower = max(lower, probe)
                    best_witness = witness
                end

                if proj.usable
                    candidate = proj.p
                    # In both-mode, never cross a rigorous separability certificate.
                    candidate_allowed = mode == :ent || candidate < upper
                    if candidate_allowed
                        ρcandidate = _mix(ρ0, ω, T(candidate))
                        candidate_val = Float64(real(dot(witness.W, ρcandidate)))
                        # A weaker projected candidate must not replace the
                        # witness certifying the already stronger lower bound.
                        if candidate > lower && candidate_val < -config.witness_tol
                            lower = max(lower, candidate)
                            best_witness = merge(witness, (
                                projected_p = candidate,
                                projected_value = candidate_val,
                                p_cross = proj.p_cross,
                            ))
                        end
                    elseif mode == :both && candidate > upper + config.bound_atol && config.verbose >= 1
                        @warn "Witness projection exceeded a certified separability upper bound" candidate upper
                    end
                end
            end

            if mode == :both
                lower <= upper || error(
                    "Certified bounds became inconsistent: ent_bound=$lower > sep_bound=$upper"
                )
            end

            # ------------------------------------------------------------------
            # Search bookkeeping.
            # ------------------------------------------------------------------
            if mode == :both
                # The left search endpoint is ALWAYS the rigorous certified lower.
                search_low = lower
                search_high = upper

                if upper_refinement
                    # Upper-only refinement: lower remains frozen/certified.  The
                    # auxiliary bracket below is purely for choosing useful probes
                    # that may tighten the rigorous geometric upper bound.
                    extrapolation = (geo !== nothing && geo.usable) ? geo.p - probe : Inf
                    if extrapolation > config.bound_atol
                        upper_refine_low = max(upper_refine_low, probe)
                    else
                        upper_refine_high = min(upper_refine_high, probe)
                    end
                    upper_refine_low = max(upper_refine_low, lower)
                    upper_refine_high = min(upper_refine_high, upper)
                    upper_refine_low = min(upper_refine_low, upper_refine_high)
                end

            elseif mode == :sep
                # Sep-only has no certified entanglement lower bound, so search_low
                # remains a purely numerical anchor inferred from the size of the
                # geometric extrapolation.
                extrapolation = (geo !== nothing && geo.usable) ? geo.p - probe : Inf
                if extrapolation > config.bound_atol
                    search_low = max(search_low, probe)
                else
                    search_high = min(search_high, probe)
                end
                search_high = min(search_high, upper)
                search_low = min(search_low, search_high)

            elseif mode == :ent
                # Ent-only also probes exclusively at the certified lower bound.
                # `search_high` has no certificate meaning here and is not used to
                # choose probes.
                search_low = lower
                search_high = 1.0
            end

            certified_gap = mode == :both ? upper - lower : NaN
            search_gap = search_high - search_low

            push!(history, (
                step = step,
                probe = probe,
                ent_bound = do_ent ? lower : NaN,
                sep_bound = do_sep ? upper : NaN,
                certified_gap = certified_gap,
                search_low = search_low,
                search_high = search_high,
                search_gap = search_gap,
                primal = Float64(res.primal),
                cg_distance = cg_distance,
                phase = (mode == :both && upper_refinement) ? :upper_refinement : :lower_refinement,
                witness_certified = ent_this_step ? witness.certified : false,
                witness_value = ent_this_step ? witness.val_rho : NaN,
                witness_eta = ent_this_step ? witness.η : NaN,
                projected_p = (ent_this_step && proj.usable) ? proj.p : NaN,
                geometric_sep = (do_sep && geo.usable) ? geo.p : NaN,
                upper_refine_low = (mode == :both && upper_refinement) ? upper_refine_low : NaN,
                upper_refine_high = (mode == :both && upper_refinement) ? upper_refine_high : NaN,
            ))

            if config.verbose >= 1
                @printf("  CG distance            = %.6e\n", cg_distance)
                if ent_this_step
                    @printf("  witness valid          = %s\n", string(witness.valid))
                    @printf("  witness detects probe  = %s\n", string(witness.detects_probe))
                    @printf("  witness certified      = %s\n", string(witness.certified))
                    @printf("  witness Tr(W rho)      = %.6e\n", witness.val_rho)
                    @printf("  witness eta            = %.12g\n", witness.η)
                    if proj.usable
                        @printf("  projected ent bound    = %.12g\n", proj.p)
                    end
                elseif mode == :both && upper_refinement
                    println("  witness refinement     = skipped (upper-only phase)")
                end
                if do_sep && geo.usable
                    @printf("  geometric sep bound    = %.12g\n", geo.p)
                end

                if mode == :both
                    @printf("  certified interval     = [%.12g, %.12g]\n", lower, upper)
                    @printf("  certified gap          = %.6e\n", certified_gap)
                    @printf("  probe-search bracket   = [%.12g, %.12g]\n", search_low, search_high)
                    @printf("  probe-search resolution= %.6e\n", search_gap)
                    if upper_refinement
                        @printf("  upper-refine bracket   = [%.12g, %.12g]\n", upper_refine_low, upper_refine_high)
                        @printf("  upper-refine resolution= %.6e\n", upper_refine_high - upper_refine_low)
                    end
                elseif mode == :sep
                    @printf("  certified sep bound    = %.12g\n", upper)
                    @printf("  search bracket         = [%.12g, %.12g]\n", search_low, search_high)
                    @printf("  search resolution      = %.6e\n", search_gap)
                else
                    @printf("  certified ent bound    = %.12g\n", lower)
                    @printf("  search bracket         = [%.12g, %.12g]\n", search_low, search_high)
                    @printf("  search resolution      = %.6e\n", search_gap)
                end
            end

            # Stopping rules.
            if mode == :both
                if certified_gap <= config.bound_atol
                    termination_status = "target_gap_reached"
                    break
                end

                progress_tol = max(10 * config.witness_margin, 1e-12)

                if upper_refinement
                    # The entanglement lower bound is frozen.  Continue trying to
                    # tighten the rigorous separability upper bound until either the
                    # certified target gap is reached or the numerical upper-search
                    # bracket itself is resolved at the requested scale.
                    upper_refine_gap = upper_refine_high - upper_refine_low
                    if upper_refine_gap <= config.bound_atol
                        termination_status = "upper_refinement_resolved_gap_remains"
                        break
                    end
                elseif step > 1 && lower <= old_lower + progress_tol
                    # Do NOT stop immediately.  Freeze the certified lower and hand
                    # the remaining iterations to an upper-only refinement phase.
                    upper_refinement = true
                    upper_refine_low = lower
                    upper_refine_high = upper

                    if config.verbose >= 1
                        println("  lower stalled          = switching to upper-only refinement")
                    end
                end

            elseif mode == :sep
                if search_gap <= config.bound_atol
                    termination_status = "sep_search_resolution_reached"
                    break
                end

            else  # :ent
                # ent-only has no rigorous upper side, so bound_atol is used as a
                # convergence tolerance on successive improvements of the certified
                # lower bound (not as an error bar on the true boundary).
                improvement = lower - old_lower
                if step > 1 && improvement <= config.bound_atol
                    termination_status = "ent_lower_improvement_below_atol"
                    break
                end
            end

            # ----- choose the next probe -----
            if mode == :sep
                probe = (search_low + search_high) / 2
            elseif mode == :both && upper_refinement
                # Entanglement refinement has ended; lower stays rigorously frozen.
                # These probes are used ONLY to improve the rigorous geometric upper.
                probe = (upper_refine_low + upper_refine_high) / 2
            else
                # Core rule in the entanglement-refinement phase: probe only a
                # rigorously certified p.
                probe = lower
            end
        end
    catch err
        err isa EntanglementDetection.TimeLimitReached || rethrow()
        termination_status = "time_limit"
    end

    certified_gap = mode == :both ? upper - lower : nothing
    search_gap = search_high - search_low

    status = if termination_status !== nothing
        termination_status
    elseif mode == :both
        if upper - lower <= config.bound_atol
            "target_gap_reached"
        elseif upper_refinement
            "max_steps_upper_refinement"
        else
            "max_steps_or_numerical_gap"
        end
    elseif mode == :sep
        (search_gap <= config.bound_atol) ? "sep_search_resolution_reached" : "max_steps_sep_search"
    else
        "max_steps_ent_search"
    end

    result = (
        mode = mode,
        ent_bound = do_ent ? lower : nothing,
        sep_bound = do_sep ? upper : nothing,
        gap = certified_gap,
        search_low = search_low,
        search_high = mode in (:both, :sep, :ent) ? search_high : nothing,
        search_gap = search_gap,
        upper_refinement_used = mode == :both ? upper_refinement : false,
        upper_refine_low = (mode == :both && upper_refinement) ? upper_refine_low : nothing,
        upper_refine_high = (mode == :both && upper_refinement) ? upper_refine_high : nothing,
        upper_refine_gap = (mode == :both && upper_refinement) ? upper_refine_high - upper_refine_low : nothing,
        status = status,
        dims = dims,
        k = k,
        structure = _structure_name(k, N),
        witness = best_witness,
        separability_certificate = best_sep,
        history = history,
        config = config,
    )

    if config.verbose >= 1
        println()
        print_bound_summary(result)
    end

    return result
end

# Pure-state convenience overload: a ket is normalized automatically.
function white_noise_bounds(
    ψ::AbstractVector,
    dims::NTuple{N, Int};
    structure = :full,
    mode::Symbol = :both,
    config::NoiseBoundConfig = NoiseBoundConfig(),
) where {N}
    n2 = real(dot(ψ, ψ))
    n2 > 0 || throw(ArgumentError("input ket has zero norm"))
    ρ = (ψ * ψ') / n2
    return white_noise_bounds(ρ, dims; structure = structure, mode = mode, config = config)
end

# Convenience one-sided wrappers.  They share exactly the same core code.
white_noise_sep_bound(args...; kwargs...) = white_noise_bounds(args...; mode = :sep, kwargs...)
white_noise_ent_bound(args...; kwargs...) = white_noise_bounds(args...; mode = :ent, kwargs...)

function print_bound_summary(result)
    println("================ FINAL WHITE-NOISE RESULT ================")
    println("mode                      : ", result.mode)
    if result.ent_bound !== nothing
        @printf("entanglement lower bound : %.12g\n", result.ent_bound)
    end
    if result.sep_bound !== nothing
        @printf("separability upper bound : %.12g\n", result.sep_bound)
    end
    if result.gap !== nothing
        @printf("certified gap             : %.6e\n", result.gap)
        if result.mode == :both && result.search_gap !== nothing
            @printf("probe-search resolution   : %.6e\n", result.search_gap)
            if result.status == "search_resolved_certificate_limited"
                println("note                      : boundary localized, but certificate gap is limited by current witness precision")
            end
        end
    else
        @printf("search resolution         : %.6e\n", result.search_gap)
        println("note                      : search bracket is not a two-sided certificate")
    end
    println("structure                 : ", result.structure)
    println("status                    : ", result.status)
    println("==========================================================")
end

end # module
