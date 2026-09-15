# ---------------------------------------------------------------------------
# Optional per-iteration trajectory recording.
#
# `results/` stores only the final bounds of a run. Some of the paper's tables
# and claims are about how a run *evolves* -- the gap-closing table averages
# over the CP iterations after the last phase begins, and the LADMM discussion
# is about how the feasibility residual behaves over iterations. Neither can be
# recovered from a result file, so both loops can emit a CSV trajectory.
#
# Set EXACTENT_TRACE to a path PREFIX; the writers append a suffix:
#
#   <prefix>.cp.csv      iter,is_last,ub_relx,lb_relx,b_lower,n_states
#   <prefix>.ladmm.csv   iter,zeta,f,pen,residual,grad_norm,z,alm
#
# In the CP trace `b_lower` is the sBB oracle's lower bound for that round, so
# lb_relx = ub_relx + b_lower. In the LADMM trace `residual` is
# ||A(z) + a - Psi(x)||_2, the quantity that certifies ub_heur when it vanishes.
#
# Tracing is off unless EXACTENT_TRACE is set, and every writer is a no-op on
# `nothing`, so the traced and untraced code paths are identical.
# ---------------------------------------------------------------------------

"""
    traceSink(suffix, header) -> IO or nothing

Open `\$EXACTENT_TRACE.<suffix>.csv` for appending, writing `header` if the file
is new. Returns `nothing` when tracing is disabled.
"""
function traceSink(suffix::AbstractString, header::AbstractString)
    prefix = get(ENV, "EXACTENT_TRACE", "")
    isempty(prefix) && return nothing
    path = string(prefix, ".", suffix, ".csv")
    mkpath(dirname(path))
    isfile(path) || open(io -> println(io, header), path, "w")
    return open(path, "a")
end

"""
    traceRow!(io, values...)

Append one CSV row. No-op when tracing is disabled.
"""
traceRow!(::Nothing, args...) = nothing
function traceRow!(io::IO, args...)
    join(io, args, ",")
    println(io)
    flush(io)   # a run that is killed by its time limit keeps what it recorded
    return nothing
end

traceClose!(::Nothing) = nothing
traceClose!(io::IO) = close(io)

"Trajectory of the cutting-plane master (paper Alg. CP)."
cpTraceSink() = traceSink("cp", "iter,is_last,ub_relx,lb_relx,b_lower,n_states")

"Trajectory of the lifted ADMM (paper Alg. LADMM)."
ladmmTraceSink() = traceSink("ladmm", "iter,zeta,f,pen,residual,grad_norm,z,alm")
