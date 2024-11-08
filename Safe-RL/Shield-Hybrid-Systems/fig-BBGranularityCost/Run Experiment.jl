if !isfile("Project.toml")
    error("Project.toml not found. Try running this script from the root of the ReproducibilityPackage folder.")
end
import Pkg
Pkg.activate(".")
Pkg.instantiate()
using Dates
include("../Shared Code/ExperimentUtilities.jl")

#########
# Args  #
#########

using ArgParse
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

    "--skip-shield-synthesis"
        help="""Skip synthesising shields. Presumes shields to be in the results dir already."""
        action=:store_true

    "--skip-evaluation"
        help="""Skip evaluation of shields. Presumes that some results already exist. Use with --skip-shield-synthesis to generate figures from existing data."""
        action=:store_true

    "--uppaal-dir"
        help="""Root directory of the UPPAAL STRATEGO 10 install."""
        default=homedir() ⨝ "opt/uppaal-4.1.20-stratego-10-linux64/"
end


args = parse_args(s)

progress_update("Time to complete is approximately 3 hours. (5 minutes with argument --test)")

# Further includes are all the way down here, to make arg error reporting faster.
using CSV
using Glob
using DataFrames
using Serialization
include("Synthesize Set of Shields.jl")
include("Get libbbshield.jl")
include("ExtractQueryResults.jl")


results_dir = args["results-dir"]
const figure_name = "fig-BBGranularityCost"

results_dir = results_dir ⨝ figure_name

# Misnomer: Also used for STRATEGO strategies.
shields_dir = results_dir ⨝ "Exported Strategies" 
mkpath(shields_dir)

query_results_dir = results_dir ⨝ "Query Results" 
mkpath(query_results_dir)

query_results_csv = query_results_dir ⨝ "Results.csv"

queries_models_dir = results_dir ⨝ "UPPAAL Queries and Models" 
mkpath(queries_models_dir)

libbbshield_dir = shields_dir
mkpath(libbbshield_dir)

libbbshield_file = libbbshield_dir ⨝ "libbbshield.so"

###
# Synthesise shields
if !args["skip-shield-synthesis"]

    if args["test"]
        # HARDCODED: Parameters to generate shield. All variations will be used.
        samples_per_axiss = [2]
        gridargss = [(0.03, -15, 15, 0, 12), (0.02, -15, 15, 0, 12)]
    else 
    
        # HARDCODED: Parameters to generate shield. All variations will be used.
        samples_per_axiss = [4]
        gridargss = [(G, -15, 15, 0, 12) for G in (0.03, 0.025, 0.02, 0.015, 0.01, 0.005)]
    end

    for file in glob("*", shields_dir)
        rm(file)
    end

    make_and_save_barbaric_shields(samples_per_axiss, gridargss, shields_dir)
else
    progress_update("Skipping synthesis of shields.")
end

# Create UPPAAL models and queries from blueprints, by doing search and replace on the placeholders.
# This is similar to templating, but the word blueprint was choseen to avoid a name clash with UPPAAL templates. 
blueprints_dir = pwd() ⨝ figure_name ⨝ "Blueprints"

if !isdir(blueprints_dir)
    throw(error("Blueprints folder not found. Make sure this script is exectued from the root of the code folder.\nCurrent directory: $(pwd())\nContents: $(readdir())"))
end

###
# Run queries and save the results to a csv.

if !args["skip-evaluation"]
    replacements = Dict(
        "%resultsdir%" => query_results_dir,
        "%shieldfile%" => libbbshield_file,
        "%checks%" => args["test"] ? 10 : 1000
    )

    search_and_replace(blueprints_dir, queries_models_dir, replacements)
    error_on_regex_in(queries_models_dir, r"%[a-zA-Z_ ]+%") # Throw error if a blueprint variable was missed


    # Backup
    query_results_backup_dir = results_dir ⨝ "Query Results Backup"

    if isdir(query_results_dir)
        if isdir(query_results_backup_dir)
            rm(query_results_backup_dir, recursive=true)
        end
        mv(query_results_dir, query_results_backup_dir)
        mkdir(query_results_dir)
    end


    # Run all queries
    abspath_model = queries_models_dir ⨝ "BB__Shielded.xml"
    abspath_query = queries_models_dir ⨝ "TrainSaveEvaluateSingle.q"

    runs = args["test"] ? 100 : 12000
    repeats = args["test"] ? 2 : 10

    results = []

    for i in 1:repeats

        mkdir(query_results_dir ⨝ "$i")

        for shield_file in glob("*.shield", shields_dir)
            
            get_libbbshield(shield_file, "Shared Code/libbbshield/", libbbshield_file, working_dir=libbbshield_dir, test=args["test"])

            shield_name = replace(basename(shield_file), ".shield" => "")

            query_results_file = query_results_dir ⨝ "$i" ⨝ "$shield_name.queryresults.txt"

            write(query_results_file, read(abspath_query))

            post_shielded_queries = ignorestatus(Cmd([
                args["uppaal-dir"] ⨝ "bin" ⨝ "verifyta",
                "-s",
                "--epsilon", "0.001",
                "--max-iterations", "1",
                "--good-runs", "$runs",
                "--total-runs", "$runs", 
                "--runs-pr-state", "$runs",
                abspath_model,
                abspath_query
            ]))

            progress_update("Running: $post_shielded_queries")

            open(query_results_file, "a") do file
                run(pipeline(post_shielded_queries, stdout=file, stderr=file))
            end

            cost, deaths, interventions = nothing, nothing, nothing
            
            try
                query_results = extract_query_results(query_results_file)

                if length(query_results) != 3
                    throw(UppaalQueryFailedException("Inconsistent number of columns"))
                end

                cost, deaths, interventions = query_results

                if parse(Float64, deaths) > 0
                    @warn "Safety violation observed for $shield_name. This should not happen."
                end

            catch ex
                if ex isa UppaalQueryFailedException
                    progress_update("Query failed. Skipping. Message: $(ex.message)")
                    continue
                else
                    throw(ex)
                end
            end

            progress_update("Adding row: $((shield_name, runs, cost, deaths, interventions))")
            push!(results, (shield_name, runs="$runs", cost, deaths, interventions))

            [mv(f, query_results_dir ⨝ "$i" ⨝ "$shield_name PreShielded.strategy.json") for f in glob("*.strategy.json", query_results_dir)]

            CSV.write(query_results_csv, DataFrame(results))
        end
    end
end

progress_update("Saving  to $results_dir")

NBPARAMS = Dict(
    "csv_file" => query_results_csv
)

include("Figure from CSV.jl")

function export_figure(name, figure)
    savefig(figure, results_dir ⨝ "$name.png")
    savefig(figure, results_dir ⨝ "$name.svg")
    progress_update("Saved $name")
end


export_figure("BBGranularityCost", granularity_cost_plot)


write(results_dir ⨝ "SafetyNotice.md", safety_notice)
progress_update("Saved SafetyNotice")

progress_update("Done with $figure_name.")
progress_update("=========================")