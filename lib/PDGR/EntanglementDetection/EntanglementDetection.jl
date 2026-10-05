# Selected source from EntanglementDetection by Y.C. Liu. See UPSTREAM.json
# and NOTICE in the parent folder for provenance and local modifications.
module EntanglementDetection
import FrankWolfe
import Ket
import LinearAlgebra as LA
import Random
import Printf
import Base.Threads

struct TimeLimitReached <: Exception end

function check_time_limit()
    deadline = get(task_local_storage(), :PDGR_deadline, Inf)
    time_ns() / 1e9 >= deadline && throw(TimeLimitReached())
    return nothing
end

include("eigmin.jl")
include("approximations.jl")
include("types.jl")
include("callback.jl")
include("fw_methods.jl")
include("utils.jl")
include("separable_distance.jl")
include("separable_ball.jl")
end
