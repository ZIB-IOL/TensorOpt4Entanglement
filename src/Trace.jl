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
#   <prefix>.cp.csv      iter,is_last,ub_relx,lb_relx,b_lower,n_states,stage
#   <prefix>.ladmm.csv   iter,zeta,f,pen,residual,grad_norm,z,alm,stage
#   <prefix>.stages.csv  stage,algorithm,event,elapsed
#
# `stage` is detector.round: IR assigns 2i-1 to LADMM and 2i to CP in
# refinement iteration i. The stage trace records start/end events even for
# calls that return without recording an iteration. `elapsed` is seconds
# since the run started; an interrupted call may have only a start event.
# Stage 0 denotes a direct call outside the IR stage numbering.
#
# In the CP trace `b_lower` is the pricing correction to that round's master
# upper bound, including any objective loss from witness stabilisation.
# `lb_relx` retains the best bound across rounds and preceding IR passes.
# In the LADMM trace `residual` is
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
    if isfile(path)
        readline(path) == header ||
            error("Trace header mismatch in $path; use a fresh EXACTENT_TRACE prefix or remove the old trace before rerunning")
    else
        open(io -> println(io, header), path, "w")
    end
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
cpTraceSink() = traceSink("cp", "iter,is_last,ub_relx,lb_relx,b_lower,n_states,stage")

"Trajectory of the lifted ADMM (paper Alg. LADMM)."
ladmmTraceSink() = traceSink("ladmm", "iter,zeta,f,pen,residual,grad_norm,z,alm,stage")

traceStageId(detector) = isnothing(detector) ? 0 : detector.round

"Record a solver call, including calls with no iteration rows."
function withTraceStage(f, algorithm::Symbol, stage::Integer, param::Param)
    trace = traceSink("stages", "stage,algorithm,event,elapsed")
    isnothing(trace) && return f()
    try
        traceRow!(trace, stage, algorithm, "start", elapsedTime(param))
        result = f()
        traceRow!(trace, stage, algorithm, "end", elapsedTime(param))
        return result
    catch
        traceRow!(trace, stage, algorithm, "error", elapsedTime(param))
        rethrow()
    finally
        traceClose!(trace)
    end
end
