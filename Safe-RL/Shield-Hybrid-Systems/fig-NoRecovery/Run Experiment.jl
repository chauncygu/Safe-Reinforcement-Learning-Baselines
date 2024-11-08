if !isfile("Project.toml")
    error("Project.toml not found. Try running this script from the root of the ReproducibilityPackage folder.")
end
import Pkg
Pkg.activate(".")
Pkg.instantiate()
using Dates
include("../Shared Code/ExperimentUtilities.jl")

# cli args
args = my_parse_args(ARGS)
if haskey(args, "help")
    print("""
    --help              Display this help and exit.
    --test              Test-mode. Produce potentially useless results, but fast. Useful for testing if everything is set up.
    --results-dir       Results will be saved in an appropriately named subdirectory. Directory will be created if it does not exist.
                        Default: '~/Results'
    """)
    exit()
end
test = haskey(args, "test")
results_dir = get(args, "results-dir", "$(homedir())/Results")
figure_name = "fig-NoRecovery"
results_dir = joinpath(results_dir, figure_name)
mkpath(results_dir)

if test
    NBPARAMS = Dict(
        "G" => 0.5,
        "samples_per_axis" => 2
    )
else
    NBPARAMS = Dict(
        "G" => 0.01,
        "samples_per_axis" => 16
    )
end

progress_update("Synthesizing safe strategies to build figure fig:NoRecovery.")
progress_update("Time to complete is approximately 50 minutes. (2 minutes with argument --test)")
    

include("BB No Recovery.jl")

progress_update("Saving to $results_dir")

# Figure saved in notebook as p1 etc...
savefig(p1, joinpath(results_dir, "$figure_name.png"))
savefig(p1, joinpath(results_dir, "$figure_name.svg"))

progress_update("Saved $figure_name")

progress_update("Done with $figure_name.")
progress_update("=========================")
