using ArgParse
using Random

using ExactEntanglement

# Entry point. Example:
#     MOSEKLM_LICENSE_FILE=/path/to/mosek.lic \
#         julia --project=. scripts/run_experiment.jl -s state_0.jl -a LADMM -t 600
function parseCommandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--state", "-s"
            help = "State file to load from benchmark directory"
            arg_type = String
            required = true
        "--algo", "-a"
            help = "algorithm (paper names): Alt-SDP, LADMM, CP, IR, DPS, DDPS+, PDGR; " *
                   "LADMM_400..LADMM_900 (m=5 rank sweep); " *
                   "DDPS, CP-DDPS, IR-DDPS (DDPS-only ablation); " *
                   "Alt-SDP+CP, IR-nolazy, IR-clear, DualALM. " *
                   "Pre-rename shorthand (A, LD1, D, LDL, PPT, RLT, LDR0-5) also accepted"
            arg_type = String
            required = true
        "--time-limit", "-t"
            help = "Time limit in seconds"
            arg_type = Float64
            default = 200.0
        "--log-level"
            help = "Log level (0-3)"
            arg_type = Int
            default = 1
        "--maxnnodes"
            help = "Maximum sBB node solves per ordinary oracle call (default: size preset)"
            arg_type = Int
        "--maxeffortnnodes"
            help = "Maximum sBB node solves per gap-closing oracle call (default: size preset)"
            arg_type = Int
        "--heur-sbb-maxiter"
            help = "Coordinate updates per eigenvector heuristic start (default: 100)"
            arg_type = Int
        "--heur-sbb-restarts"
            help = "Random eigenvector heuristic starts before the sBB root (default: 1)"
            arg_type = Int
        "--heur-sbb-node-restarts"
            help = "Eigenvector heuristic starts guided by each node relaxation (default: 4)"
            arg_type = Int
        "--minnnodes"
            help = "Minimum number of nodes"
            arg_type = Int
            default = 1
        "--maxrounds"
            help = "Maximum number of separation rounds"
            arg_type = Int
            default = 100
        "--heur-ladmm1-maxiter"
            help = "LADMM outer iterations in the first IR pass (default: size preset)"
            arg_type = Int
        "--heur-ladmm-maxiter"
            help = "LADMM outer iterations after the first IR pass (default: size preset)"
            arg_type = Int
        "--heur-manopt-maxiter"
            help = "Manopt iterations per inner solve (default: size preset; doubled for standalone LADMM)"
            arg_type = Int
        "--heur-ladmm-penalty-update"
            help = "LADMM penalty update: balance or legacy (default: size/algorithm preset)"
            arg_type = String
        "--heur-ladmm-conjugates"
            help = "Add conjugate product columns at the LADMM-to-CP crossover for real targets"
            arg_type = Bool
            default = false
        "--cp-real-master"
            help = "Real CP master with conjugate-pair reconstruction (default: size/algorithm preset)"
            arg_type = Bool
        "--cp-rounds-per-ir"
            help = "CP rounds per intermediate IR pass (-1 disables the cap; default: size/algorithm preset)"
            arg_type = Int
        "--cp-certify-every"
            help = "Stabilise the witness and certify at the sBB root every N CP rounds (0 disables; default: preset)"
            arg_type = Int
        "--ir-refit-scalar"
            help = "Refit the scalar after CP rank trimming (default: size/algorithm preset)"
            arg_type = Bool
        "--relaxation"
            help = "sBB relaxation: ddpsplus (DDPS + McCormick, default) or ddps"
            arg_type = String
            default = "ddpsplus"
        "--seed"
            help = "RNG seed"
            arg_type = Int
            default = 12345
        "--loop"
            help = "LP loops"
            arg_type = Int
            default = -1
        "--pdgr-mode"
            help = "PDGR certificate mode: both, ent, or sep"
            arg_type = String
            default = "both"
        "--pdgr-bound-atol"
            help = "PDGR target bound gap (default 1e-3)"
            arg_type = Float64
        "--pdgr-max-steps"
            help = "PDGR maximum noise probes (default 15)"
            arg_type = Int
        "--pdgr-fw-epsilon"
            help = "PDGR Frank-Wolfe primal tolerance (default 1e-7)"
            arg_type = Float64
        "--pdgr-fw-max-iteration"
            help = "PDGR maximum Frank-Wolfe iterations per probe (default 1000000)"
            arg_type = Int
        "--pdgr-lmo-nb"
            help = "PDGR alternating oracle restarts (default 10)"
            arg_type = Int
        "--pdgr-lmo-max-iter"
            help = "PDGR iterations per alternating oracle restart (default 1000)"
            arg_type = Int
        "--pdgr-witness-max-length"
            help = "PDGR witness net size budget (default 10000000)"
            arg_type = Int
    end
    return parse_args(s)
end

function main()
    println("Starting ExactEntanglement.jl main script...")
    args = parseCommandline()

    returnval = runEntangle(args)

    return returnval
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
