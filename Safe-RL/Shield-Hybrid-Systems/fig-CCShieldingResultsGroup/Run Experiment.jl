if !isfile("Project.toml")
    error("Project.toml not found. Try running this script from the root of the ReproducibilityPackage folder.")
end
import Pkg
pkg_project_path = abspath(".")
Pkg.activate(pkg_project_path)
Pkg.instantiate()

using ArgParse
using Glob
using Dates
include("../Shared Code/ExperimentUtilities.jl")
include("../Shared Code/Get libccshield.jl")

s = ArgParseSettings()

# infix operator "\join" redefined to signify joinpath
⨝ = joinpath

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
            help="""Root directory of the UPPAAL STRATEGO 11 install."""
            # Stratego 10 has a bug related to the Cruise-control example, while the final release of Stratego 11 has a bug that prevents running models with .q files in CLI.
            default=homedir() ⨝ "opt/uppaal-4.1.20-stratego-11-rc1-linux64/"

        "--julia-dir"
            help="""Root directory of the julia install. Used to locate the file <julia-dir>/share/julia/julia_config.jl"""
            default=dirname(Base.julia_cmd()[1]) ⨝ ".."

        "--skip-experiment"
            help="""Yea I know. But figures will still be created from <results-dir>/Query Results/Results.csv
                    If nothing else I need this for testing."""
            action=:store_true
end

args = parse_args(s)

progress_update("Estimated total time to commplete: 34 hours. (10 minutes if run with --test)")

results_dir = args["results-dir"]
const figure_name = "fig-CCShieldingResultsGroup"
results_dir = results_dir ⨝ figure_name

queries_models_dir = results_dir ⨝ "UPPAAL Queries and Models"
mkpath(queries_models_dir)

query_results_dir = results_dir ⨝ "Query Results"
mkpath(query_results_dir)

libccshield_working_dir = results_dir ⨝ "libcc"
mkpath(libccshield_working_dir)

# julia-config.jl provides args to gcc for when the julia library is a dependency.
# See https://docs.julialang.org/en/v1/manual/embedding/
julia_config_dir=args["julia-dir"] ⨝ "share/julia/julia-config.jl"

if !isfile(julia_config_dir)
    error("Not found: julia-config.jl\nLooked in: $julia_config_dir\nPlease provide a vallid --julia-dir")
end

possible_shield_file = args["shield"] #results_dir ⨝ "../tab-BBSynthesis/Exported Strategies/400 Samples 0.01 G.shield"

checks = args["test"] ? 10 : 1000 # Number of checks to use for estimating expected outcomes in the UPPAAL queries

if !args["skip-experiment"]
    # Get the nondeterministic safe strategy that will be used for shielding.
    # Or just the "shield" for short.
    libccpreshield_file = libccshield_working_dir ⨝ "libccpreshield.so"
    libccpostshield_file = libccshield_working_dir ⨝ "libccpostshield.so"
    strategy_for_postshield = libccshield_working_dir ⨝ "postshieldme.strategy.json" # Path where strategies will be loaded by UPPAAL models and post-shielded. All Queries.py will copy strategies to here.

    postshield_notebook_path = abspath(figure_name ⨝ "PostShield Strategy.jl")
    source_code_dir = "Shared Code/libccshield/"    # Destination of the C files used to compile the libcc binaries

    # Get the raw shield file first
    shield_file = get_shield(possible_shield_file, libccshield_working_dir, test=args["test"])

    # Then bake it into the libcc binaries
    get_libccshield(shield_file, source_code_dir, 
                        preshield_destination=libccpreshield_file, 
                        postshield_destination=libccpostshield_file, 
                        working_dir=libccshield_working_dir, 
                        julia_config_dir=julia_config_dir,
                        test=args["test"])

    # Create UPPAAL models and queries from blueprints, by doing search and replace on the placeholders.
    # This is similar to templating, but the word blueprint was choseen to avoid a name clash with UPPAAL templates. 
    blueprints_dir = pwd() ⨝ figure_name ⨝ "Blueprints" # TODO: $figure_name/Blueprints

    if !isdir(blueprints_dir)
        throw(error("Blueprints folder not found. Make sure this script is exectued from the root of the code folder.\nCurrent directory: $(pwd())\nContents: $(readdir())"))
    end

    replacements = Dict(
        "%resultsdir%" => query_results_dir,                # Where the .q files should save the learnedstrategies
        "%libccpreshield_path%" => libccpreshield_file,     # Where the libccpreshield.so file is located
        "%libccpostshield_path%" => libccpostshield_file,   # Where the libccposthield.so file is located
        "%checks%" => checks,                               # Used in queries: E[<=120;checks] (max:...)
        
        # These are passed from the PostShielded UPPAAL models to postshield.c
        "%postshield_notebook_path%" => postshield_notebook_path,   # Location of the Pluto Notebook that contains the post-shielding code
        "%strategy_for_postshield%" => strategy_for_postshield,     # Where said notebook should load the strategy from
        "%shield_for_postshield%" => shield_file,                   # Where said notebook should load the shield from
        "%pkg_project_path%" => pkg_project_path,                   # Path to the julia packages "project" to ensure that PostShield Strategy.jl is loaded with the correct packages.
    )

    search_and_replace(blueprints_dir, queries_models_dir, replacements)
    error_on_regex_in(queries_models_dir, r"%[a-zA-Z_ ]+%")

    # I don't recall why I wrote this particular code in python.
    # I think it was because I knew how to use python's os.system() but not julia's run().
    # And as you can see, Julia's run() is kind of strange. https://docs.julialang.org/en/v1/manual/running-external-programs/

    @assert isdir(args["uppaal-dir"])
    cmd = [
        "python3", figure_name ⨝ "All Queries.py", 
        "--results-dir", query_results_dir,
        "--strategy-for-postshield", strategy_for_postshield,
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
    "checks" => checks,
    "default_post_shield_type" => "PostShieldedRandomChoice"
)

include("ReadResults.jl")


export_figure(results_dir, "CCShieldingResults", average_cost)

export_figure(results_dir, "CCShieldingInterventions", average_interventions)

export_figure(results_dir, "CCShieldingDeaths", average_deaths)

#export_figure(results_dir, "CCPostShieldCostComparison", post_shield_cost_bar)

#export_figure(results_dir, "CCPostShieldInterventionsComparison", post_shield_interventions_bar)

if safety_violations !== nothing
    if !args["test"]
        progress_update("WARNING: Safety violation observed in shielded configuration. This is unexpected.")
    else
        progress_update("Safety violation observed in shielded configuration. This may not be unexpected, since the experiment was run as --test.")
    end
else
    progress_update("No deaths observed in pre-shielded or post-shielded models.")
end

write(results_dir ⨝ "SafetyNotice.md", safety_violations_message)
progress_update("Saved SafetyNotice")

write(results_dir ⨝ "CCPostShieldComparison.tex", percent_change_from_random′)
progress_update("Saved CCPostShieldComparison")

progress_update("Done with $figure_name.")
progress_update("====================================")
