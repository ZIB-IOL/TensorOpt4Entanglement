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
            help = "algorithm (paper names): Alt-SDP, LADMM, CP, IR, DPS, DDPS+; " *
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
            help = "Maximum number of nodes"
            arg_type = Int
            default = 100
        "--minnnodes"
            help = "Minimum number of nodes"
            arg_type = Int
            default = 1
        "--maxrounds"
            help = "Maximum number of separation rounds"
            arg_type = Int
            default = 100
        "--heur-ladmm1-maxiter"
            help = "Heuristic LADMM1 maximum iterations"
            arg_type = Int
            default = 16
        "--heur-ladmm-maxiter"
            help = "Heuristic LADMM maximum iterations"
            arg_type = Int
            default = 8
        "--heur-manopt-maxiter"
            help = "Heuristic MANOPT maximum iterations"
            arg_type = Int
            default = 150
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