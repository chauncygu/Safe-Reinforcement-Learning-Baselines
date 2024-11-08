### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 8617b0cb-b308-4fe2-9a23-d7b4823b2f9d
begin
	using Pkg
	if isdefined(Main, :PlutoRunner) # Check if run as notebook
		Pkg.activate("..")
	end
	using GridShielding

	include("../Shared Code/DC-DC Converter.jl")
	include("../Shared Code/DCShielding.jl")
	include("../Shared Code/FlatUI.jl");
	include("../Shared Code/ExperimentUtilities.jl")
	
	using CSV
	using Plots
	using Dates
	using PlutoUI
	using Setfield
	using Serialization
	TableOfContents()
end

# ╔═╡ 96aad91c-38c6-11ed-2211-bd576258e89c
md"""
# Synthesize Set of Shields

Synthesize and export a set of shields from different given parameters.
"""

# ╔═╡ fdbdf588-101d-4c12-940a-55072186ade6
m = DCMechanics()

# ╔═╡ 09dca934-5613-4f77-a2e9-048bcc4dc26d
call(f) = f()

# ╔═╡ 011370e4-e238-4f3a-a409-b9e6daf08417
md"""
## Generating a shield and checking if its valid
"""

# ╔═╡ 8213b25c-483a-4fbc-ac99-5adc103eadf0
# ╠═╡ skip_as_script = true
#=╠═╡
md"""

`granularity =` $(@bind granularity NumberField(0.001:0.001:1, default=0.1))

"""
  ╠═╡ =#

# ╔═╡ 6b2933a2-ab85-47a0-945c-6043da25e7b6
#=╠═╡
grid = get_dc_grid(m, granularity)
  ╠═╡ =#

# ╔═╡ 6a78d494-ce7b-437c-a529-ef245f4d5e71
randomness_space = get_randomness_space(m)

# ╔═╡ a13bc371-4748-4afa-988f-8b421935de91
# ╠═╡ skip_as_script = true
#=╠═╡
md"""

Select number of samples per axis. 

The location variable $p$ can only ever have 1 sample per axis, since it can only take on values 0 and 1.

$(@bind samples_per_axis_selected NumberField(1:30, default=2))
"""
  ╠═╡ =#

# ╔═╡ 6ff9144c-64f6-484f-9b37-03eedce6b964
#=╠═╡
samples_per_axis = [samples_per_axis_selected, samples_per_axis_selected, 1, samples_per_axis_selected]
  ╠═╡ =#

# ╔═╡ 8b166982-79fd-49bf-8d76-302c2383682d


# ╔═╡ 087973b4-4366-4968-a918-704a6d635689
simulation_function = get_simulation_function(m)

# ╔═╡ 041f227a-30c1-4bb1-949a-a9af31180cff
#=╠═╡
simulation_model = SimulationModel(simulation_function, randomness_space, samples_per_axis)
  ╠═╡ =#

# ╔═╡ ab7f32ab-106e-4b8b-8c36-e7333bccaae3
#=╠═╡
R̂ = get_barbaric_reachability_function(simulation_model)
  ╠═╡ =#

# ╔═╡ 5da4304b-6c0f-4de6-90e5-0358aeb6d9d7
slice = [:, :, 1, 1]

# ╔═╡ f7535470-76f6-4f90-ae5d-c684c28dc0cd
#=╠═╡
R̂_precomputed = get_transitions(R̂, SwitchStatus, grid);
  ╠═╡ =#

# ╔═╡ 0984a1e6-df91-4112-aaef-d63f427650fa
# ╠═╡ skip_as_script = true
#=╠═╡
shield, _ = make_shield(R̂_precomputed, SwitchStatus, grid)
  ╠═╡ =#

# ╔═╡ fc158839-cd54-464c-bd7f-3fc97257ea93
#=╠═╡
shield_is_valid(shield)
  ╠═╡ =#

# ╔═╡ 4255b905-92f5-4e6b-a744-47eb9b9eb73e
#=╠═╡
begin
	draw(shield, slice,
		legend=:outerright, 
		colors=opshieldcolors,
		color_labels=opshieldlabels;
		xlabel="t", ylabel="v")
end
  ╠═╡ =#

# ╔═╡ 87a9bb3a-bdc7-4769-b930-b4697efc0736
md"""
## Code for batch-generating shields

Generate multiple shields based on different reachability function parameters and different grid sizes.
"""

# ╔═╡ 815eb159-b92a-4de8-953f-fc5c8aeed323
# Sample output for @timed macro:
@timed begin
	f(a::BigInt) = a > 1 ? a*f(a-1) : 1
	f(a::Number) = f(BigInt(a))
	f(300)
end

# ╔═╡ 061630a8-828c-4136-8c66-c71fe96a1784
#=╠═╡
simulation_model
  ╠═╡ =#

# ╔═╡ 8523418d-67e4-4d02-9697-46e274e3e61e
function make_and_save_barbaric_shield(samples_per_axis, granularity, save_path)
	simulation_model′ = SimulationModel(simulation_function, randomness_space, samples_per_axis)
	
	# Prerequesites for synthesizing shield
	R̂ = get_barbaric_reachability_function(simulation_model′)
	
	grid = get_dc_grid(m, granularity)
	
	# Mainmatter
	_, seconds_taken, bytes_used, gctime, gcstats = @timed begin
		shield, max_steps_reached = make_shield(R̂, SwitchStatus, grid)
	end

	max_steps_reached && @warn "Max steps reached! This should not happen."

	# Save shield
	sample_count = length(possible_outcomes(simulation_model′, 
			first(shield), off))
	
	file_name = "DC $sample_count Samples $(granularity[1]) G.shield"
	file_path = get_unique_file(save_path, file_name)
	robust_grid_serialization(file_path, shield)
	
	plot = draw(shield, slice,
		legend=:outerright, 
		colors=opshieldcolors,
		color_labels=opshieldlabels;
		xlabel="t", ylabel="v")
	
	savefig(plot, file_path * ".png")

	return (saved_as=basename(file_path), 
		samples_per_axis, 
		gridargs=(shield.granularity, shield.bounds.lower, shield.bounds.upper),
		valid=shield_is_valid(shield), 
		seconds_taken, 
		bytes_used, 
		gctime, 
		gcstats.allocd, gcstats.malloc, gcstats.realloc, gcstats.poolalloc, gcstats.bigalloc, gcstats.freecall, gcstats.total_time, gcstats.pause, gcstats.full_sweep)
end

# ╔═╡ a8e4ffec-98a4-4eb1-8d61-22c2d2e729f2
# ╠═╡ skip_as_script = true
#=╠═╡
test_dir = mktempdir(prefix="jl_synthesis_")
  ╠═╡ =#

# ╔═╡ 451aa409-aff6-4104-8108-44600971fad0
# ╠═╡ skip_as_script = true
#=╠═╡
barbaric_shield = make_and_save_barbaric_shield(samples_per_axis, granularity, test_dir)
  ╠═╡ =#

# ╔═╡ 6cd2dced-fab7-4ef0-a72c-8cb58d773935
#=╠═╡
draw(shield, slice,
		legend=:outerright, 
		colors=opshieldcolors,
		color_labels=opshieldlabels;
		xlabel="t", ylabel="v")
  ╠═╡ =#

# ╔═╡ 1ea060e2-d10c-454a-b6e3-85f54796c793
function estimate_time(samples_per_axis, G)
	# These time estimates are not from DC, but they are probably within the same order of magnitude
	
	# Time to compute R̂ with 45 samples and 968688 squares
	secnods_per_sample = 2991/(135*968688)
	
	# Time to do fixed-point computation over 968688 squares
	additional_seconds_per_square = 1377/968688

	# Number of samples per square
	samples = prod(samples_per_axis) # 3 axes and 5 possible actions

	# Number of squares
	lower, upper = 
		[m.x1_min, m.x2_min - G[2], m.R_min],
		[m.x1_max + G[1], m.x2_max + G[2], m.R_max + 1]

	
	grid_size = zeros(Int, 3)
	for (i, (lb, ub)) in enumerate(zip(lower, upper))
		grid_size[i] = ceil((ub-lb)/G[i])
	end
	
	squares = prod(grid_size)

	return squares*samples*secnods_per_sample + squares*additional_seconds_per_square
end

# ╔═╡ 11752808-f613-4082-aad4-2b12be33c7eb
#=╠═╡
estimate_time(samples_per_axis, [granularity, granularity, 1])
  ╠═╡ =#

# ╔═╡ 15c07be8-c422-407c-94d4-ebe6829ae6cc
# Notice the "s" at the end of the name. This is a case where argument types can't save me I think.
function estimate_times(samples_per_axiss, Gs)
	sum(estimate_time(samples_per_axis, G) 
			for samples_per_axis in samples_per_axiss
			for G in Gs)
end

# ╔═╡ 84351342-3c6a-491a-bde3-ef9a90b7ef14
function make_and_save_barbaric_shields(samples_per_axiss, Gs, save_path)
	if !isdir(save_path)
		throw(ArgumentError("Not a directory: $save_path"))
	end

	nshields = length(samples_per_axiss)*length(Gs)
	progress_update("The following number of shields will be synthesized: $nshields")
	estimated_seconds = estimate_times(samples_per_axiss, Gs)
	estimated_minutes = estimated_seconds/60
	progress_update("Estimated time to complete: $estimated_minutes minutes")
	
	results = []
	for samples_per_axis in samples_per_axiss
		for G in Gs
			result = make_and_save_barbaric_shield(samples_per_axis, G, save_path)
			push!(results, result)
			
			progress_update("Shield $(length(results)) done. Valid:$(result.valid). Saved as: $(result.saved_as)")
		end
	end

	report_path = get_unique_file(save_path, "Barbaric Shields Synthesis Report.csv")
	CSV.write(report_path, results)
	
	return results
end

# ╔═╡ Cell order:
# ╟─96aad91c-38c6-11ed-2211-bd576258e89c
# ╠═fdbdf588-101d-4c12-940a-55072186ade6
# ╠═8617b0cb-b308-4fe2-9a23-d7b4823b2f9d
# ╠═09dca934-5613-4f77-a2e9-048bcc4dc26d
# ╟─011370e4-e238-4f3a-a409-b9e6daf08417
# ╠═8213b25c-483a-4fbc-ac99-5adc103eadf0
# ╠═6b2933a2-ab85-47a0-945c-6043da25e7b6
# ╠═6a78d494-ce7b-437c-a529-ef245f4d5e71
# ╟─a13bc371-4748-4afa-988f-8b421935de91
# ╠═6ff9144c-64f6-484f-9b37-03eedce6b964
# ╠═8b166982-79fd-49bf-8d76-302c2383682d
# ╠═087973b4-4366-4968-a918-704a6d635689
# ╠═041f227a-30c1-4bb1-949a-a9af31180cff
# ╠═ab7f32ab-106e-4b8b-8c36-e7333bccaae3
# ╠═5da4304b-6c0f-4de6-90e5-0358aeb6d9d7
# ╠═f7535470-76f6-4f90-ae5d-c684c28dc0cd
# ╠═0984a1e6-df91-4112-aaef-d63f427650fa
# ╠═fc158839-cd54-464c-bd7f-3fc97257ea93
# ╟─4255b905-92f5-4e6b-a744-47eb9b9eb73e
# ╟─87a9bb3a-bdc7-4769-b930-b4697efc0736
# ╠═815eb159-b92a-4de8-953f-fc5c8aeed323
# ╠═061630a8-828c-4136-8c66-c71fe96a1784
# ╠═8523418d-67e4-4d02-9697-46e274e3e61e
# ╠═a8e4ffec-98a4-4eb1-8d61-22c2d2e729f2
# ╠═451aa409-aff6-4104-8108-44600971fad0
# ╟─6cd2dced-fab7-4ef0-a72c-8cb58d773935
# ╠═84351342-3c6a-491a-bde3-ef9a90b7ef14
# ╠═11752808-f613-4082-aad4-2b12be33c7eb
# ╠═1ea060e2-d10c-454a-b6e3-85f54796c793
# ╠═15c07be8-c422-407c-94d4-ebe6829ae6cc
