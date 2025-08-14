using Test
using ExactEntanglement

# Construct args dict manually for testing
function create_test_args(;
    state::String = "state_0.jl",
    algo::String = "LDL",
    time_limit::Float64 = 1000.0,
    log_level::Int = 1,
    maxnnodes::Int = 1000,
    minnnodes::Int = 1,
    maxrounds::Int = 100,
    heur_ad_depth::Int = 5,
    heur_ad_maxiters::Int = 100,
    heur_ladmm1_maxiter::Int = 16,
    heur_ladmm_maxiter::Int = 8,
    heur_manopt_depth::Int = 3,
    heur_manopt_maxiter::Int = 200,
    loop::Int = -1
)
    return Dict{String, Any}(
        "state" => state,
        "algo" => algo,
        "time-limit" => time_limit,
        "log-level" => log_level,
        "maxnnodes" => maxnnodes,
        "minnnodes" => minnnodes,
        "maxrounds" => maxrounds,
        "heur-ad-depth" => heur_ad_depth,
        "heur-ad-maxiters" => heur_ad_maxiters,
        "heur-ladmm1-maxiter" => heur_ladmm1_maxiter,
        "heur-ladmm-maxiter" => heur_ladmm_maxiter,
        "heur-manopt-depth" => heur_manopt_depth,
        "heur-manopt-maxiter" => heur_manopt_maxiter,
        "loop" => loop
    )
end

@testset "Main script integration" begin
    # Prepare arguments as would be passed from the command line

    success = false
    result = nothing

    # Call the main function
    args = create_test_args()
    println("Running main script with args: $args")
    result = runEntangle(args)
    success = true

end