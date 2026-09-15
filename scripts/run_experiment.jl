using ArgParse
using Random

using ExactEntanglement

# Entry point. Example:
#     MOSEKLM_LICENSE_FILE=/path/to/mosek.lic \
#         julia --project=. scripts/run_experiment.jl -s state_0.jl -a LD1 -t 600
function parseCommandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--state", "-s"
            help = "State file to load from benchmark directory"
            arg_type = String
            required = true
        "--algo", "-a"
            help = "algorithm: LD/LD0/LD1/LDL (iterative refinement), LDR0-LDR5 " *
                   "(LADMM rank sweep, m=5), D (cutting plane), A (alternating SDP), " *
                   "AD (alternating SDP + cutting plane), PPT (DPS bound), " *
                   "RLT (DDPS+ bound), LDual (experimental dual ALM)"
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