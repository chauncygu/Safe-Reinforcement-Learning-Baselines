if !isfile("Project.toml")
    error("Project.toml not found. Try running this script from the root of the ReproducibilityPackage folder.")
end
import Pkg
Pkg.activate(".")
Pkg.instantiate()
using ArgParse
⨝ = joinpath # infix operator "\join" redefined to signify joinpath

#########
# Args  #
#########
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

        "--uppaal-dir"
            help="""Root directory of the UPPAAL STRATEGO 10 install."""
            default=homedir() ⨝ "opt/uppaal-4.1.20-stratego-10-linux64/"

        "--skip-barbaric"
            help="""Skip synthesis using barbaric reachability funciton."""
            action=:store_true

        "--skip-rigorous"
            help="""Skip synthesis using rigorous reachability funciton."""
            action=:store_true

        "--skip-synthesis"
            help="""Skip shield synthesis entirely. Equivalent to --skip-barbaric --skip-rigorous"""
            action=:store_true

        "--skip-evaluation"
            help="""Do not evaluate the strategies' safety after synthesis is done."""
            action=:store_true
end

args = parse_args(s)

test = args["test"]
results_dir = args["results-dir"]
table_name = "tab-BBSynthesis"
results_dir = joinpath(results_dir, table_name)
shields_dir = joinpath(results_dir, "Exported Strategies")
uppaal_dir = args["uppaal-dir"]
mkpath(shields_dir)
evaluations_dir = joinpath(results_dir, "Evaluations")
mkpath(evaluations_dir)

make_barbaric_shields = !args["skip-barbaric"] && !args["skip-synthesis"]
make_rigorous_shields = !args["skip-rigorous"] && !args["skip-synthesis"]
test_shields = !args["skip-evaluation"]

#########
# Setup #
#########

# Additional usings after arguments are parse, to make cli help and error reporting slightly faster. Every bit counts because it is abysmally slow.
using Dates
using Serialization
using Glob
include("../Shared Code/ExperimentUtilities.jl")
include("../Shared Code/Get libbbshield.jl")
include("Synthesize Set of Shields.jl")
include("CheckSafetyOfPreshielded.jl")

progress_update("Estimated total time to complete: 75 hours. (5 minutes if run with --test. 4 hours if run with --skip-rigorous, 71 hours if run with --skip-barbaric)")

if !test
    # HARDCODED: Parameters to generate shield. All variations will be used.
    samples_per_axiss = [1, 2, 3, 4, 8, 16]
    barbaric_gridargss = [(0.05, -15, 15, 0, 12), (0.02, -15, 15, 0, 12), (0.01, -15, 15, 0, 12)]

    # HARDCODED: Parameters to generate shield. All variations will be used.
    # algorithms = [
	# 	AlgorithmInfo(BOX(δ=0.01), 4, 
	# 		"BOX 0.01"),
	# 	AlgorithmInfo(BOX(δ=0.002), 20, 
	# 		"BOX 0.002"),
	# 	AlgorithmInfo(GLGM06(δ=0.01, max_order=10, approx_model=Forward()), 9,
	# 		"GLGM06 0.01"), 
	# 	AlgorithmInfo(GLGM06(δ=0.002, max_order=10, approx_model=Forward()), 11,
	# 		"GLGM06 0.002"),
    # ]
    # rigorous_gridargss = [(0.02, -15, 15, 0, 14), (0.01, -15, 15, 0, 12)]
    # Here is a set of parameters that should be able to finish over night
    algorithms = [
        AlgorithmInfo(BOX(δ=0.002), 20, "BOX 0.002"),
        AlgorithmInfo(BOX(δ=0.001), 30, "BOX 0.001"),
    ]
    rigorous_gridargss = [(0.01, -15, 15, 0, 12)]

    # HARDCODED: Safety checking parameters.
    random_agents_hit_chances = [1/10]
    runs_per_shield = 1000000
else 
    # Test params that produce uninteresting results quickly
    samples_per_axiss = [2]
    barbaric_gridargss = [(0.1, -15, 15, 0, 12), (0.02, -15, 15, 0, 12)]

    algorithms = [
		AlgorithmInfo(BOX(δ=0.01), 4, 
			"BOX 0.01")
    ]
    rigorous_gridargss = [(0.5, -15, 15, 0, 12)]

    random_agents_hit_chances = [1/10]
    runs_per_shield = 100
end

##############
# Mainmatter #
##############

if make_barbaric_shields
    progress_update("Estimated time: $(test ? 60 : 3863) seconds")
    make_and_save_barbaric_shields(samples_per_axiss, barbaric_gridargss, shields_dir)
else
    progress_update("Skipping synthesis of shields using sampling-based reachability analysis.")
end

if make_rigorous_shields
    make_and_save_rigorous_shields(algorithms, rigorous_gridargss, shields_dir)
else
    progress_update("Skipping synthesis using ReachabilityAnalysis.jl package.")
end

if test_shields
    check_safety_of_preshielded(;shields_dir, results_dir=evaluations_dir, lib_source_code_dir="Shared Code/libbbshield",  blueprints_dir="tab-BBSynthesis/Blueprints", uppaal_dir, test, just_print_the_commands=false)
else
    progress_update("Skipping tests of shields")
end

progress_update("Computation done.")

# The make_and_save*shields methods produce two seperate tables
# because my code is as near unworkable spaghetti as it is possible without prompting a rewrite.
# So I gotta merge the two. Let's go. 
using CSV
using DataFrames

barbric_df, rigorous_df = nothing, nothing

barbaric_file = joinpath(shields_dir, "Barbaric Shields Synthesis Report.csv")

if isfile(barbaric_file)
    barbric_df = 
        open(barbaric_file) do file
            CSV.read(file, DataFrame)	
        end
end

rigorous_file = joinpath(shields_dir, "Rigorous Shields Synthesis Report.csv")

if isfile(rigorous_file)
    rigorous_df = 
        open(rigorous_file) do file
            CSV.read(file, DataFrame)
        end
end

joint_df = append!(something(barbric_df, DataFrame()),
                 something(rigorous_df, DataFrame()),
                 cols=:union)

joint_df |> CSV.write(joinpath(shields_dir, "Joint Shields Synthesis Report.csv"))



######################
# Constructing Table #
######################

progress_update("Saving  to $results_dir")

NBPARAMS = Dict(
    "csv_synthesis_report" => joinpath(shields_dir, "Joint Shields Synthesis Report.csv"),
    "csv_safety_report" => joinpath(evaluations_dir, "Evaluations.csv")
)

include("Table from CSVs.jl")

exported_table_name = "BBSynthesis"

CSV.write(joinpath(results_dir, "$exported_table_name.csv"), joint_report)
write(joinpath(results_dir, "$exported_table_name.txt"), "$joint_report")
write(joinpath(results_dir, "$exported_table_name.tex"), "$resulting_latex_table")

# Oh god this is so hacky. These macros are used in the paper so I have to define them here also.
write(joinpath(results_dir, "macros.tex"), 
"""\\newcommand{\\granularity}{G}
\\newcommand{\\state}{s}
\\newcommand{\\juliareach}{\\textsc{JuliaReach}\\xspace}""")


progress_update("Saved $(exported_table_name)")

progress_update("Done with $table_name.")
progress_update("====================================")
