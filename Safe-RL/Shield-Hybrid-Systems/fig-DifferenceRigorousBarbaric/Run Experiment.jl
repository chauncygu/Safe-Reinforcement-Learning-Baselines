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

shield1_default = homedir() ⨝ "Results/tab-BBSynthesis/Exported Strategies/4 Samples 0.01 G.shield"
shield2_default = homedir() ⨝ "Results/tab-BBSynthesis/Exported Strategies/BOX 0.001 with G of 0.01.shield"

@add_arg_table s begin
    "--results-dir"
        help="""Results will be saved in an appropriately named subdirectory.
                Directory will be created if it does not exist."""            
        default=homedir() ⨝ "Results"

    "--shield1"
        help="""First shield file to use for the experiment."""
        default=shield1_default

    "--shield2"
        help="""Second shield file to use for the experiment."""
        default=shield2_default

    "--color-mode"
        help="""Color theme to use for the differences. One of {transparent, distinctive}."""
        default="transparent"
end

args = parse_args(s)

results_dir = args["results-dir"]
const figure_name = "fig-DifferenceRigorousBarbaric"
results_dir = results_dir ⨝ figure_name

mkpath(results_dir)

# Additional includes to make arg parsing go through faster
using Plots
using Dates
using Serialization
include("../Shared Code/FlatUI.jl")
include("../Shared Code/PlotsDefaults.jl")
include("../Shared Code/ExperimentUtilities.jl")
include("../Shared Code/BBSquares.jl")

diffcolors = args["color-mode"] == "distinctive" ? [
	colors.SUNFLOWER,  # {hit, nohit} ~ {}
	colors.PETER_RIVER,  # {hit} ~ {}
	colors.AMETHYST, # {hit, nohit} ~ {hit}
	colors.WET_ASPHALT # {hit, nohit} ~ {hit}
] :  args["color-mode"] == "transparent" ? [
	colorant"#717171", # {hit, nohit} ~ {}
	colorant"#FCF434", # {hit} ~ {}
	colorant"#ff48ff", # {hit, nohit} ~ {hit}
] : error("--color-mode should be one of {distinctive, transparent}")

function error_on_missing(file::AbstractString)
    if !isfile(file)
        error("Could not find file $file. This experiment is dependent on data from tab-BBSynthesis.")
    end
end

error_on_missing(args["shield1"])
error_on_missing(args["shield2"])

shield1 = robust_grid_deserialization(args["shield1"])
shield2 = robust_grid_deserialization(args["shield2"])

function get_descriptor(filename)
	if occursin("BOX", filename)
		return "BOX"
	elseif occursin(r"N|samples|Samples", filename)
		return "Barbaric"
	else
		return replace(filename, ".shield" => "")
	end
end

name1 = get_descriptor(args["shield1"])
name2 = get_descriptor(args["shield2"])


p1 = draw_diff(shield1, shield2, diffcolors, bbshieldcolors, bbshieldlabels; name1, name2,
    # plotargs
    xlabel="Velocity (m/s)", ylabel="Position (m)", legend_position=:outertop)

plot!(size=(300, 300), 
    xlims=(-10, 7), 
    ylims=(3, 7.5))

const name = "DifferenceRigorousBarbaric"

savefig(p1, results_dir ⨝ "$name.png")
savefig(p1, results_dir ⨝ "$name.svg")
progress_update("Saved $name")


