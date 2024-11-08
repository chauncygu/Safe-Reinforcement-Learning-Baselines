if !isfile("Project.toml")
    error("Project.toml not found. Try running this script from the root of the ReproducibilityPackage folder.")
end

using Pkg
Pkg.activate(".")
Pkg.instantiate()
using ArgParse

# infix operator "\join" redefined to signify joinpath
⨝ = joinpath

figure_name = "fig-BBShieldRobustness"

s = ArgParseSettings()

@add_arg_table s begin
    "--test"
    help = """Test-mode. Produce potentially useless results, but fast.
              Useful for testing if everything is set up."""
    action = :store_true

    "--results-dir"
    help = """Results will be saved in an appropriately named subdirectory.
              Directory will be created if it does not exist."""
    default = homedir() ⨝ "Results"

    "--shield"
    help = """Shield file to use for the experiment. 
              If no file is provided, a new shield will be synthesised and saved in the results dir."""
    default = nothing

    "--skip-experiment"
    help="""Yea I know. But figures will still be created from <results-dir>/Results.csv
            If nothing else I need this for testing."""
    action=:store_true
end

args = parse_args(s)

# Remaining usings after ArgParse to speed up error reporting.
using Glob
using CSV
using Dates
using Plots
using ProgressLogging
include("Get libbbshield.jl")
include("Check Robustness of Shields.jl")
include("../Shared Code/ExperimentUtilities.jl")


results_dir = args["results-dir"]
results_dir = results_dir ⨝ figure_name
mkpath(results_dir)

possible_shield_file = args["shield"]

β1 = bbmechanics.β1
hit_chance = 1/10

if args["test"]
    β1s = β1-0.08:0.01:β1+0.03
    runs_per_configuration = 100
else
    β1s = β1-0.08:0.005:β1+0.03
    runs_per_configuration = 100000
end


if !args["skip-experiment"]

    shield_file = get_shield(possible_shield_file, results_dir, test=args["test"])

    shield = robust_grid_deserialization(shield_file)

    progress_update("Testing the shield's robustness...")
    progress_update("Estimated total time to complete: 1 hour. (1 minute if run with --test)")
    
    robustness_results = check_robustness_of_shield(shield, bbmechanics, β1s, hit_chance, runs_per_configuration)
    
    write(results_dir ⨝ "RawResults.txt", "$robustness_results")
    CSV.write(results_dir ⨝ "RawResults.csv", robustness_results)
    progress_update("Saved RawResults")

end


progress_update("Saving  to $results_dir")

p1 = robustness_plot(results_dir ⨝ "RawResults.csv")

robustness_plot_name = "BBShieldRobustness"
savefig(p1, results_dir ⨝ "$robustness_plot_name.png")
savefig(p1, results_dir ⨝ "$robustness_plot_name.svg")
progress_update("Saved $robustness_plot_name")


progress_update("Done with $figure_name.")
progress_update("====================================")
