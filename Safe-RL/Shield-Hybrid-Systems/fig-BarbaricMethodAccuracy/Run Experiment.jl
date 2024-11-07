if !isfile("Project.toml")
    error("Project.toml not found. Try running this script from the root of the ReproducibilityPackage folder.")
end
import Pkg
Pkg.activate(".")
Pkg.instantiate()


using ArgParse
s = ArgParseSettings()

# infix operator "\join" redefined to signify joinpath
⨝ = joinpath

@add_arg_table s begin
    "--test"
    help = """Test-mode. Produce potentially useless results, but fast.
              Useful for testing if everything is set up."""
    action = :store_true

    "--results-dir"
        help="""Results will be saved in an appropriately named subdirectory.
                Directory will be created if it does not exist."""            
        default=homedir() ⨝ "Results"

    "--skip-experiment"
    help="""Yea I know. But figures will still be created from <results-dir>/Results.csv
            If nothing else I need this for testing."""
    action=:store_true
end

args = parse_args(s)


const figure_name = "fig-BarbaricMethodAccuracy"

results_dir = args["results-dir"]

mkpath(results_dir)

# Additional includes here to make arg parsing go through faster
using CSV
using Dates
using DataFrames
include("Reliability of Barbaric Method.jl")
include("../Shared Code/ExperimentUtilities.jl")


results_dir = joinpath(results_dir, figure_name)
mkpath(results_dir)

if !args["test"]
    squares_to_test = 100000
    samples_per_square = 1000

    # To use with the spa comparison
    granularities1 = [1, 0.01]
    samples_per_axiss1 = [2:16;] # Values of `samples_per_axis` to test for
    
    # To use with the granularities comparison
    granularities2 = [1, 0.5, 0.25, 0.1, 0.05, 0.04, 0.02, 0.01]
    samples_per_axiss2 = [4, 8]
else
    squares_to_test = 100
    samples_per_square = 100

    # To use with the spa comparison
    granularities1 = [1, 0.01]
    samples_per_axiss1 = [2:16;] # Values of `samples_per_axis` to test for
    
    # To use with the granularities comparison
    granularities2 = [1, 0.5, 0.25, 0.1, 0.05, 0.04, 0.02, 0.01]
    samples_per_axiss2 = [4, 8]
end

estimated_time = estimate_time(granularities1, samples_per_axiss1, samples_per_square, squares_to_test)
estimated_time += estimate_time(granularities2, samples_per_axiss2, samples_per_square, squares_to_test)
progress_update("Estimated time to complete: $estimated_time seconds.")

if !args["skip-experiment"]
    spa_df = compute_accuracies(granularities1, samples_per_axiss1, bbmechanics; samples_per_square, squares_to_test)
    granularity_df = compute_accuracies(granularities2, samples_per_axiss2, bbmechanics; samples_per_square, squares_to_test)

    # Save as csv, txt
    export_table(results_dir, "BarbaricAccuracyN", spa_df)
    export_table(results_dir, "BarbaricAccuracyG", granularity_df)

    progress_update("Computation done.")
else 
    spa_df = CSV.read(results_dir ⨝ "BarbaricAccuracyN.csv", DataFrame)
    granularity_df = CSV.read(results_dir ⨝ "BarbaricAccuracyG.csv", DataFrame)
end

progress_update("Saving  to $results_dir")

p1 = plot_accuracies_spa(spa_df)
p2 = plot_accuracies_granularity(granularity_df)

export_figure(results_dir, "BarbaricAccuracyN", p1)
export_figure(results_dir, "BarbaricAccuracyG", p2)

progress_update("Done with $figure_name.")
progress_update("====================================")