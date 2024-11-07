### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 4f2b3a13-0160-45a1-930f-7028af1bb4e2
begin
	using Pkg
	if isdefined(Main, :PlutoRunner)
		Pkg.activate("..")
	end
	using GridShielding

	include("../Shared Code/DC-DC Converter.jl")
	include("../Shared Code/DCShielding.jl")
	include("../Shared Code/FlatUI.jl")
	include("../Shared Code/ExperimentUtilities.jl")
	
	using CSV
	using Glob
	using Plots
	using Dates
	using PlutoUI
	using Setfield
	using StatsBase
	using Serialization
	TableOfContents()
end

# ╔═╡ 1c3007e6-35b5-11ed-14d9-4bd1323de584
md"""
# Statistical Checking of Shield
"""

# ╔═╡ 894124f6-12bd-4d38-9a58-fd6791fca109
call(f) = f()

# ╔═╡ df61eece-81c4-48eb-8465-742f92593982
m = DCMechanics()

# ╔═╡ facedf11-2020-412e-afdc-425e2bf292fc
slice = [:, :, 1, 1]

# ╔═╡ fc7fda02-669f-4c95-b1cb-b6a638727f44
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
## (action required) Import Shield

`selected_file` = $(@bind selected_file PlutoUI.FilePicker([MIME("application/octet-stream")]))
"""
  ╠═╡ =#

# ╔═╡ ed05be2d-abe3-484c-9d47-a1ca90812445
#=╠═╡
if selected_file == nothing
	shield = nothing
	md"""
!!! danger "Please select file"
	"""
else
	shield = robust_grid_deserialization(selected_file["data"] |> IOBuffer)
	
	
	draw(shield, slice,
		legend=:outerright, 
		colors=opshieldcolors,
		color_labels=opshieldlabels;
		xlabel="t", ylabel="v")
end
  ╠═╡ =#

# ╔═╡ b4d72971-1a4c-4270-80b5-a5ae8d18ebf5
#=╠═╡
shield
  ╠═╡ =#

# ╔═╡ 03127da2-5690-4202-9950-5b1d5a55acaa
#=╠═╡
shield_is_valid(shield)
  ╠═╡ =#

# ╔═╡ 96717e9a-0acb-4c8c-a237-2dbcb7fb85af
md"""
## Functions to Check a Policy

Shield will be checked on a series of random policies. 
Each random policy is defined by its chance at any given time-step of swinging at the ball.

The number of safety violations are recorded, and the last unsafe trace will be saved.
"""

# ╔═╡ 18870590-acc3-462b-8b45-3e90f82d9a77
function get_random_policy(off_chance)
	random_agent(_...) = sample([on, off], [1 - off_chance, off_chance] |> Weights)
end

# ╔═╡ 5a4a59db-21f0-4ad3-b3ee-d7d14f2a74e0
#=╠═╡
call() do 
	random_policy = get_random_policy(0.5)
	shielded_random_policy = shielded(shield, random_policy)
	
	anim = @animate for i in 1:30
		trace = simulate_trace(m, initial_state, shielded_random_policy, duration=120)
		plot(ylims=(0, 30))
		plot_trace!(trace, show_actions=false)

		hline!([m.x1_max, m.x2_min, m.x2_max], 
			color=colors.WET_ASPHALT,
			label="safety constraints")
	end
	gif(anim, joinpath(tempdir(), "gif3.gif"), show_msg=false, fps=2)
end
  ╠═╡ =#

# ╔═╡ 65dfc820-e7be-4bc2-bae7-14f4991625a8
function test_policies(m::DCMechanics, 
		policy_functions, 
		policy_labels, 
		number_of_runs)	
	
	result = []
	for (policy, policy_label) in zip(policy_functions, policy_labels)
		
		safety_violations_observed, _, unsafe_trace = count_unsafe_traces(m, 
			policy, 
			runs=number_of_runs, 
			run_duration=120)
		
		push!(result, 
			(;policy_label, safety_violations_observed, number_of_runs, unsafe_trace))
	end	
	result
end

# ╔═╡ 35f23b12-a2dd-42d9-be85-5c4c30b5c229
begin
	"""
		test_shield(shield::Grid, number_of_runs::Number)
	
		
		test_shield(path_to_shield::AbstractString, number_of_runs::Number)

	Test shield on random agents with different "hit" frequency.
	"""
	function test_shield(m::DCMechanics, shield::Grid, number_of_runs::Number)

		if !shield_is_valid(shield)
			progress_update("This is not a valid shield. Skipping.")
			return [(policy_label="n/a", safety_violations_observed=1, number_of_runs=1, unsafe_trace=[])]
		end
		
		random_policy = get_random_policy(0.5)
		shielded_random_policy = shielded(shield, random_policy)
		
		policy_functions = [shielded_random_policy]
		policy_labels = ["shielded random agent"]

		
		result = test_policies(m, policy_functions, policy_labels, number_of_runs)
		
		if any([run.safety_violations_observed > 0 for run in result])
			progress_update("A safety violation was observed for the shield.")
		else
			progress_update("No safety violations observed for shield.")
		end
		
		result
	end

	
	function test_shield(m::DCMechanics, path_to_shield::AbstractString, number_of_runs::Number)
		if !isfile(path_to_shield)
			throw(ArgumentError("Shield not found at path: $path_to_shield"))
		end
	
		shield = robust_grid_deserialization(path_to_shield)
		# Policies observed to be unsafe.
		# List of tuples (label, safety_violations_observed, runs, unsafe_trace)
		return test_shield(m, shield, number_of_runs)
	end
end

# ╔═╡ b5ec75ff-24e7-4b0d-8a5c-0bf1ae5225a0
#=╠═╡
if shield != nothing
	test_shield(m, shield, 100)
end
  ╠═╡ =#

# ╔═╡ 965455ec-079a-4fad-a841-613ee7bf71b6
@bind path_to_shield TextField((80, 1), placeholder="path to shield", default="/home/asger/Results/tab-DCSynthesis/Exported Strategies/DC samples 4 granularity 0.1.shield")

# ╔═╡ 31a98389-5c53-49e1-866b-1d2a614d3062
# ╠═╡ skip_as_script = true
#=╠═╡
test_shield(m, path_to_shield, 100)
  ╠═╡ =#

# ╔═╡ e062ce52-8ddd-4f00-b7bc-e6d60a5b3828
md"""
## Functions to Save Results to Disk
"""

# ╔═╡ 958522e6-8310-4336-88bb-e5c2cdf83eaf
"""
	test_multiple_shields_and_save_results(shields_dir, results_dir, 
		random_agents_hit_chances, runs_per_shield)

Load shields from `shields_dir` and test them on a random agent. 

Number of safety violations are counted and saved in `results_dir` as one big csv.

Additionally, the `rawdata` files are created for each run, and will contain an example of an unsafe trace if one was found.
"""
function test_shields_and_save_results(m::DCMechanics, 
		shields_dir, 
		results_dir, 
		runs_per_shield)
	
	result = []
	for path_to_shield in glob("*.shield", shields_dir)
		name = basename(path_to_shield)
		progress_update("Checking shield: $name")
		
		output = test_shield(m, path_to_shield, runs_per_shield)

		output = [merge((file=name, ), o)
					for o in output]

		serialize(get_unique_file(results_dir, "Test of $name.rawresults"), output)
		result = [result; output]
	end
		

	csv_data = [
		(file, label, safety_violations_observed, runs, 
			fraction_unsafe=safety_violations_observed/runs)
		for (file, label, safety_violations_observed, runs, _) in result
	]
	
	CSV.write(get_unique_file(results_dir, "Test of Shields.csv"), csv_data)
	result
end

# ╔═╡ 1afde70b-4558-4948-b01b-bff68b195ad9
# ╠═╡ skip_as_script = true
#=╠═╡
test_dir = mktempdir(prefix="jl_checking_")
  ╠═╡ =#

# ╔═╡ 166e7950-3612-4e9a-a8fd-a2f6d3e79ee8
# ╠═╡ skip_as_script = true
#=╠═╡
test_shields_and_save_results(m,
	dirname(path_to_shield), 
	test_dir, 
	100)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─1c3007e6-35b5-11ed-14d9-4bd1323de584
# ╠═4f2b3a13-0160-45a1-930f-7028af1bb4e2
# ╠═894124f6-12bd-4d38-9a58-fd6791fca109
# ╠═df61eece-81c4-48eb-8465-742f92593982
# ╠═facedf11-2020-412e-afdc-425e2bf292fc
# ╟─fc7fda02-669f-4c95-b1cb-b6a638727f44
# ╠═b4d72971-1a4c-4270-80b5-a5ae8d18ebf5
# ╠═03127da2-5690-4202-9950-5b1d5a55acaa
# ╟─ed05be2d-abe3-484c-9d47-a1ca90812445
# ╟─96717e9a-0acb-4c8c-a237-2dbcb7fb85af
# ╠═18870590-acc3-462b-8b45-3e90f82d9a77
# ╠═b5ec75ff-24e7-4b0d-8a5c-0bf1ae5225a0
# ╟─5a4a59db-21f0-4ad3-b3ee-d7d14f2a74e0
# ╠═65dfc820-e7be-4bc2-bae7-14f4991625a8
# ╠═35f23b12-a2dd-42d9-be85-5c4c30b5c229
# ╠═965455ec-079a-4fad-a841-613ee7bf71b6
# ╠═31a98389-5c53-49e1-866b-1d2a614d3062
# ╟─e062ce52-8ddd-4f00-b7bc-e6d60a5b3828
# ╠═958522e6-8310-4336-88bb-e5c2cdf83eaf
# ╠═1afde70b-4558-4948-b01b-bff68b195ad9
# ╠═166e7950-3612-4e9a-a8fd-a2f6d3e79ee8
