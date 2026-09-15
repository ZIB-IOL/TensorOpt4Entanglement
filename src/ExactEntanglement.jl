"""
    ExactEntanglement

Global optimization over separable density tensors (SDTO).

Implements the algorithms of the accompanying paper:

| paper                                  | here                      |
|----------------------------------------|---------------------------|
| LADMM  (Alg. "lifted ADMM")            | `ladmmSolve`              |
| CP     (Alg. "cutting-plane")          | `cuttingPlane`            |
| IR     (Alg. "iterative refinement")   | `iterativeRefinement`     |
| sBB LMO (Sect. "branch-and-bound LMO") | `separate!` / `threshold!`|

Layout: `sbb/` is the branch-and-bound oracle, `cuttingplane/` the master
problem, `solvers/` the lifted nonconvex solvers, and the three files at the
top level are shared foundations.

`runEntangle` is the entry point used by `scripts/run_experiment.jl`; see
`ALGORITHMS` in `Drivers.jl` for the `-a` codes.

Naming: this package uses `lowerCamelCase` for its own functions and
`UpperCamelCase` for types. The `snake_case` methods (`manifold_dimension`,
`zero_vector!`, `retract_project!`, ...) are interface methods whose names are
fixed by ManifoldsBase/Manopt.
"""
module ExactEntanglement

# ---- solver / modelling -------------------------------------------------
using JuMP
import MathOptInterface as MOI
import Mosek
import Ket as Ket

# Mosek is an optional runtime dependency: the package must still load (and the
# pure-Julia heuristics must still run) on a machine without a Mosek licence.
try
    using MosekTools
catch e
    @warn "MosekTools unavailable; solver-backed algorithms will fail" exception = e
end

# ---- numerics -----------------------------------------------------------
using LinearAlgebra
using Random
using DataStructures
using TensorOperations

# ---- manifold optimisation ----------------------------------------------
using Zygote
using Manifolds, ManifoldsBase
using Manopt
using RecursiveArrayTools
import ManifoldsBase:
    representation_size, manifold_dimension, inner,
    zero_vector, zero_vector!, retract_project!,
    parallel_transport_to!, log!, rand!, copy
import Manopt: max_stepsize, get_reason, get_solver_return

# ---- foundations --------------------------------------------------------
include("Types.jl")        # Status, Param, run clock
include("MathUtils.jl")    # index maps, McCormick helpers, small numerics
include("Solver.jl")       # Mosek setup and result classification
include("Lift.jl")         # the smooth lift Psi and its gradient
include("Trace.jl")        # optional per-iteration trajectory recording

# ---- sBB linear-minimisation oracle -------------------------------------
include("sbb/Tree.jl")          # bipartition tree
include("sbb/Node.jl")          # search node and its variable bounds
include("sbb/Problem.jl")       # separation problem + model container
include("sbb/Separator.jl")     # search state, pruning, global bounds
include("sbb/Relaxations.jl")   # convex relaxations and valid inequalities
include("sbb/Branching.jl")     # branching rule and node selection
include("sbb/BoundTighten.jl")  # optimisation-based bound tightening
include("sbb/Heuristics.jl")    # primal heuristics
include("sbb/Search.jl")        # the oracle itself: separate! / threshold!
include("Diagnostics.jl")       # relaxation size and peak memory

# ---- cutting-plane master -----------------------------------------------
include("cuttingplane/Detector.jl")           # abstract detector + state pool
include("cuttingplane/CuttingPlane.jl")       # the CP algorithm
include("cuttingplane/ThresholdDetector.jl")  # threshold detector + drivers

# ---- lifted nonconvex solvers -------------------------------------------
include("solvers/LADMM.jl")
include("solvers/AlternatingSDP.jl")
include("solvers/DualALM.jl")

# ---- command-line driver ------------------------------------------------
include("Drivers.jl")

export Param
export runEntangle
export detectEntanglementThresholdLiftDiscrete,
       detectEntanglementThresholdDiscrete,
       detectEntanglementThresholdHybridSingle

end # module ExactEntanglement
