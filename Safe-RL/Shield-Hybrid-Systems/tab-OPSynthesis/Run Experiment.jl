if !isfile("Project.toml")
    error("Project.toml not found. Try running this script from the root of the ReproducibilityPackage folder.")
end
import Pkg
Pkg.activate(".")
Pkg.instantiate()


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

    "--uppaal-dir"
        help="""Root directory of the UPPAAL STRATEGO 10 install."""
        default=homedir() ⨝ "opt/uppaal-4.1.20-stratego-10-linux64/"

    "--skip-synthesis"
        help="""Skip synthesising shields. Presumes shields to be in the results dir already."""
        action=:store_true

    "--skip-evaluation"
        help="""Skip evaluation of shields. Presumes that some results already exist. Use with --skip-shield-synthesis to generate figures from existing data."""
        action=:store_true
end

args = parse_args(s)
test = args["test"]
results_dir = args["results-dir"]
table_name = "tab-OPSynthesis"
results_dir = joinpath(results_dir, table_name)
shields_dir = joinpath(results_dir, "Exported Strategies")
mkpath(shields_dir)
evaluations_dir = joinpath(results_dir, "Evaluations")
mkpath(evaluations_dir)
uppaal_dir = args["uppaal-dir"]
@assert isdir(uppaal_dir) uppaal_dir
make_barbaric_shields = !args["skip-synthesis"]
test_shields = !args["skip-evaluation"]


#########
# Setup #
#########


using Dates
include("../Shared Code/ExperimentUtilities.jl")
include("CheckSafetyOfPreshielded.jl")
include("OP Synthesize Set of Shields.jl")
include("OP Statistical Checking of Shield.jl")

if !test
    # HARDCODED: Parameters to generate shield. All variations will be used.
    samples_per_axiss = [1, 2, 3, 4]
    Gs = [1, 0.5, 0.2, 0.1]

    # HARDCODED: Safety checking parameters.
    runs_per_shield = 1E6
else 
    # Test params that produce uninteresting results quickly
    samples_per_axiss = [1]
    Gs = [1, 2]
    
    runs_per_shield = 100
end


# samples per axis and granularities are individually defined for each axis. The 3rd axis, p, only has discrete values 0 and 1, and therefore should not be sampled at other points.
samples_per_axiss = [ (s, s, 1, s) for s in samples_per_axiss ]
Gs = [ (G, G, 1, G) for G in Gs]

m = OPMechanics()


##############
# Mainmatter #
##############


progress_update("Estimated total time to complete: 3 hours. (2 minutes if run with --test.)")

if make_barbaric_shields
    make_and_save_barbaric_shields(m, samples_per_axiss, Gs, shields_dir)
else
    progress_update("Skipping synthesis of shields using sampling-based reachability analysis.")
end

if test_shields
    check_safety_of_preshielded(;shields_dir, results_dir=evaluations_dir, lib_source_code_dir="Shared Code/libopshield", blueprints_dir="tab-OPSynthesis/Blueprints", uppaal_dir, test, just_print_the_commands=false)
else
    progress_update("Skipping tests of shields")
end


######################
# Constructing Table #
######################


NBPARAMS = Dict(
    "csv_synthesis_report" => joinpath(shields_dir, "Barbaric Shields Synthesis Report.csv"),
    "csv_safety_report" => joinpath(evaluations_dir, "Evaluations.csv")
)


###########
# Results #
###########



progress_update("Saving  to $results_dir")

include("Table from CSVs.jl")

exported_table_name = "OPSynthesis"

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
