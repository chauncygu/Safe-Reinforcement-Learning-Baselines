if !isfile("Project.toml")
    error("Project.toml not found. Try running this script from the root of the ReproducibilityPackage folder.")
end

using Pkg
Pkg.activate(".")
Pkg.instantiate()
using ArgParse
using Glob
using Dates
include("../Shared Code/ExperimentUtilities.jl")
include("../Shared Code/Get libopshield.jl")

# infix operator "\join" redefined to signify joinpath
⨝ = joinpath

figure_name = "fig-OPShieldingResultsGroup"

s = ArgParseSettings()

@add_arg_table s begin
    "--test"
        help="""Test-mode. Produce potentially useless results, but fast.
                Useful for testing if everything is set up."""
        action=:store_true

        "--results-dir"
            help="""Results will be saved in an appropriately named subdirectory.
                    Directory will be created if it does not exist."""
            default=homedir() ⨝ "Results"

        "--shield"
            help="""Shield file to use for the experiment. 
                    If no file is provided, a new shield will be synthesised and saved in the results dir."""
            default=nothing

        "--uppaal-dir"
            help="""Root directory of the UPPAAL STRATEGO 10 install."""
            default=homedir() ⨝ "opt/uppaal-4.1.20-stratego-10-linux64/"

        "--skip-experiment"
            help="""Yea I know. But figures will still be created from <results-dir>/Query Results/Results.csv
                    If nothing else I need this for testing."""
            action=:store_true
end

args = parse_args(s)

results_dir = args["results-dir"]
results_dir = results_dir ⨝ figure_name

queries_models_dir = results_dir ⨝ "UPPAAL Queries and Models"
mkpath(queries_models_dir)

query_results_dir = results_dir ⨝ "Query Results"
mkpath(query_results_dir)

libopshield_dir = results_dir ⨝ "libopshield"
mkpath(libopshield_dir)

possible_shield_file = args["shield"]

checks = args["test"] ? 10 : 1000 # Number of checks to use for estimating expected outcomes in the UPPAAL queries

if !args["skip-experiment"]
    # Get the nondeterministic safe strategy that will be used for shielding.
    # Or just the "shield" for short.
    libopshield_file = libopshield_dir ⨝ "libopshield.so"
    get_libopshield(possible_shield_file, "Shared Code/libopshield/", libopshield_file, working_dir=libopshield_dir, test=args["test"])


    # Create UPPAAL models and queries from blueprints, by doing search and replace on the placeholders.
    # This is similar to templating, but the word blueprint was choseen to avoid a name clash with UPPAAL templates. 
    blueprints_dir = pwd() ⨝ figure_name ⨝ "Blueprints"

    if !isdir(blueprints_dir)
        throw(error("Blueprints folder not found. Make sure this script is exectued from the root of the code folder.\nCurrent directory: $(pwd())\nContents: $(readdir())"))
    end

    replacements = Dict(
        "%resultsdir%" => query_results_dir,
        "%shieldfile%" => libopshield_file,
        "%checks%" => checks
    )
        
    search_and_replace(blueprints_dir, queries_models_dir, replacements)
        
    # I don't recall why I wrote this particular code in python.
    # I think it was because I knew how to use python's os.system() but not julia's run().
    # And as you can see, Julia's run() is kind of strange. https://docs.julialang.org/en/v1/manual/running-external-programs/

    cmd = [
        "python3", figure_name ⨝ "All Queries.py", 
        "--results-dir", query_results_dir,
        "--queries-models-dir", queries_models_dir,
        "--uppaal-dir", args["uppaal-dir"],
    ]

    if args["test"]
        push!(cmd, "--test")
    end

    cmd = Cmd(cmd)

    progress_update("Starting up Python script 'All Queries.py'")

    run(`echo $cmd`)

    Base.exit_on_sigint(false)

    try
        run(cmd)
    catch ex
        if isa(ex, InterruptException)
            # Couldn't figure out how to kill by handle lol
            # If you're using other python apps or something, that's tough.
            println("Interrupt Handling: Killing child processes using killall. \n(And whichever other process are unlucky enough to share their names 💀)")
            killcommand = `killall python3`
            run(`echo $killcommand`)
            run(killcommand, wait=false)
            killcommand = `killall verifyta`
            run(`echo $killcommand`)
            run(killcommand, wait=false)
        end
        throw(ex)
    end
    progress_update("Computation done.")
end

progress_update("Saving  to $results_dir")

NBPARAMS = Dict(
    "selected_file" => results_dir ⨝ "Query Results/Results.csv",
    "layabout" => false,
    "checks" => checks
)

include("ReadResults.jl")

average_cost_name = "OPShieldingResults"
savefig(average_cost, results_dir ⨝ "$average_cost_name.png")
savefig(average_cost, results_dir ⨝ "$average_cost_name.svg")
progress_update("Saved $average_cost_name")

average_interventions_name = "OPShieldingInterventions"
savefig(average_interventions, results_dir ⨝ "$average_interventions_name.png")
savefig(average_interventions, results_dir ⨝ "$average_interventions_name.svg")
progress_update("Saved $average_interventions_name")

average_deaths_name = "OPShieldingDeaths"
savefig(average_deaths, results_dir ⨝ "$average_deaths_name.png")
savefig(average_deaths, results_dir ⨝ "$average_deaths_name.svg")
progress_update("Saved $average_deaths_name")

write(results_dir ⨝ "SafetyNotice.md", safety_violations_message)
if safety_violations !== nothing
    if !args["test"]
        progress_update("WARNING: Safety violation observed in shielded configuration. This is unexpected.")
    else
        progress_update("Safety violation observed in shielded configuration. This may not be unexpected, since the experiment was run as --test.")
    end
else
    progress_update("No deaths observed in pre-shielded or post-shielded models.")
end

progress_update("Done with $figure_name.")
progress_update("====================================")