### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ f474fbb6-900b-4e18-8ba0-12a85358d077
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using Plots
	using PlutoUI
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ bdb61d68-2ac9-11ed-1b20-9dc154fc45d0
begin
	using JSON
	using Random
	using StatsBase
	#using Unzip
	using Serialization
	using Colors
	include("../Shared Code/FlatUI.jl")
	include("../Shared Code/Cruise.jl")
	include("../Shared Code/Squares.jl")
	include("../Shared Code/ShieldSynthesis.jl")
end

# ╔═╡ 12556d17-f66f-4db3-8718-3576b8a2c8dc
call(f) = f()

# ╔═╡ c2d7aa5e-70b3-4c9c-8196-7f07959e9802
strategycolors=[colors.GREEN_SEA, colors.WET_ASPHALT, colors.BELIZE_HOLE]

# ╔═╡ 5660755c-7418-4ffd-9e57-0f1a5d168fef
md"""
# The resulting functions:

This notebook provides the functions `get_shielded_strategy_int` and `get_intervention_checker` to the file `postshield.c`

	deterministic_shielded_strategy = get_shielded_strategy_int(strategy_path, shield_path, true)

	nondeterministic_shielded_strategy = get_shielded_strategy_int(strategy_path, shield_path, false)

	intervened = get_intervention_checker(strategy_path, shield_path)
"""

# ╔═╡ 93288f99-4efd-46c8-ab2b-93432a4ddd06
#=╠═╡
md"""
`v_ego =` $(@bind v_ego NumberField(-10:2:20, default=0))

`v_front =`  $(@bind v_front NumberField(-10:2:20, default=0))

`distance =` $(@bind distance NumberField(-1:1:201, default=10))
"""
  ╠═╡ =#

# ╔═╡ 2dde1bfc-f79d-4936-9a82-6006c550e2ee
[(actions_to_int(CCAction, [x]), x) for x in [backwards, neutral, forwards]]

# ╔═╡ c41edb3e-7051-4ea6-a1cd-636f89fe9483
md"""
# Creating a Shielded Strategy

Select the two files containing a UPPAAL STRATEGO strategy and a shield, to get the shielded strategy. This is useful for post-shielding which cannot be done natively in UPPAAL. (yet.)

In addition to the post-shielded strategy function, a function is provided to indicate whether the shield intervenes at any given state.

All policy choices are wrapped in a `[]` because there should be the possibility to return multiple actions to choose from when the original policy does not itself choose a safe action.
"""

# ╔═╡ eeebe049-8b0b-4e3e-8cb2-4ee89e241273
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
**Exported UPPAAL STRATEGO strategy:** 

`selected_file` = $(@bind selected_file PlutoUI.FilePicker([MIME("application/json")]))
"""
  ╠═╡ =#

# ╔═╡ c870988d-96aa-4114-9901-35472f341d16
#=╠═╡
if selected_file == nothing
	md"""
!!! danger "Error"
	# Please select file
"""
end
  ╠═╡ =#

# ╔═╡ fd0135aa-78c8-49cb-9ee5-dc1599881d16
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
**Exported Shield:** 

`shield_file` = $(@bind shield_file PlutoUI.FilePicker([MIME("application/octet-stream")]))
"""
  ╠═╡ =#

# ╔═╡ 2abe9b5e-8673-4627-bca0-5d2b4940eb27
#=╠═╡
if shield_file == nothing
	md"""
!!! danger "Error"
	# Please select file
"""
end
  ╠═╡ =#

# ╔═╡ 8894e942-b6f6-4893-bfea-ebaa2f9c58f0
md"""
## Importing the Strategy:
"""

# ╔═╡ 9a3af469-7dc0-4d29-a375-bdc79415e950
#=╠═╡
jsondict = selected_file["data"] |> IOBuffer |> JSON.parse
  ╠═╡ =#

# ╔═╡ 3eb461a0-6f98-43a7-986a-423afb55766c
#=╠═╡
regressors = jsondict["regressors"]["(1)"]["regressor"]
  ╠═╡ =#

# ╔═╡ 667f2a9f-633e-43d4-9421-dd31338995f2
#=╠═╡
regressor = regressors["1"]
  ╠═╡ =#

# ╔═╡ b3e2a24a-fab6-414e-be8b-351a0e7d1b1c
md"""
### Methods to Read the Strategy File

Please verify that the action map is correct.
"""

# ╔═╡ 4d616f3f-3bb9-4a14-b785-838786849a4d
#=╠═╡
jsondict["actions"]
  ╠═╡ =#

# ╔═╡ bef24a24-6229-497f-8d55-2b51478289e6
function edge_to_ccaction(edge::AbstractString)
	if match(r"accelerationEgo := -2", edge) != nothing
		return backwards
		
	elseif match(r"accelerationEgo := 0", edge) != nothing
		return neutral
		
	elseif match(r"accelerationEgo := 2", edge) != nothing
		return forwards
	else
		return nothing # Should never come up.
	end
end

# ╔═╡ 35e3c5a6-39e1-4f92-94af-09ff6148b358
function get_action_map(jsondict)
	return Dict(
		(k => edge_to_ccaction(v)) 
		for (k, v) in jsondict["actions"]
		if edge_to_ccaction(v) != nothing
	)
end

# ╔═╡ 293c4fa8-9fe2-4fd7-ad4f-bbdcc2e0614b
#=╠═╡
action_map = get_action_map(jsondict)
  ╠═╡ =#

# ╔═╡ 0da72168-bc4d-41e0-865a-a29ddd762ddf
edge_to_ccaction("Ego.Choose->Ego.Positive_acc { velocityEgo < maxVelocityEgo, tau, accelerationEgo := 2 }")

# ╔═╡ 5204d97e-6e04-4525-818b-a7ed642aec0d
function get_predicted_outcome(regressor, state)
	# Base case
	if typeof(regressor) <: Number
		return regressor
	end

	# Recursion
	var_index = regressor["var"] + 1 # Julia indexes start at 1
	var = state[var_index]
	bound = regressor["bound"]
	if var >= bound
		get_predicted_outcome(regressor["high"], state)
	else
		get_predicted_outcome(regressor["low"], state)
	end
end

# ╔═╡ 2cdfba1e-b6f2-4e9f-82fd-da14ef7415f5
#=╠═╡
[k for (k, _) in regressors]
  ╠═╡ =#

# ╔═╡ 7f840af3-bff8-4548-9d31-cf3c64fa9faf
function get_action(regressors, action_map, state; allowed=instances(CCAction))
	# The regressor uses strings to represent actions.
	regressor_actions = [k for (k, _) in regressors]

	lowest_outcome = Inf
	cheapest_action = nothing
	for regressor_action in regressor_actions
		outcome = get_predicted_outcome(regressors[regressor_action], state)
		action = action_map[regressor_action]
		if outcome < lowest_outcome && action ∈ allowed
			lowest_outcome = outcome
			cheapest_action = action
		end
	end
	[cheapest_action]
end

# ╔═╡ f969579d-e392-4dce-88ba-6a5da83b599f
#=╠═╡
function draw(policy::Function, x_min, x_max, y_min, y_max, G, slice; plotargs...)

	# Wrap policy in actions_to_int()
	policy_int(s) = begin
		result = actions_to_int(CCAction, policy(s))
	end
	
	if 2 != count((==(Colon())), slice)
		throw(ArgumentError("The slice argument should be an array of indices and exactly two colons. Example: [:, 10, :]"))
	end

	# Build matrix
	ix, iy = indexof((==(Colon())), slice)
	
	size_x, size_y = Int((x_max - x_min)/G), Int((y_max - y_min)/G)
	matrix = Matrix(undef, size_x, size_y)
	for i in 1:size_x
		for j in 1:size_y
			x, y = i*G - G + x_min, j*G - G + y_min
			state = [v for v in slice]
			state[ix] = x
			state[iy] = y
			matrix[i, j] = Int(policy_int(state))
		end
	end
	matrix[1,1] = 0b000
	matrix[1,2] = 0b111 # HACK: Need to contain min and max values
	x_tics = G+x_min:G:x_max
	y_tics = G+y_min:G:y_max

	# Print labels
	labels = [x for x in 0:length(ccshieldcolors)-1]
	labels = [lpad(string(x, base=2), 3, '0') for x in labels]
	
	plot(;plotargs...)
	for (l, c) in zip(labels, ccshieldcolors)
		plot!(Float64[], Float64[], seriestype=:shape, 
			legend=:outerright,
			color=c, label=l)
	end
	heatmap!(x_tics, y_tics,
		    transpose(matrix),
			color=ccshieldcolors,
			colorbar=nothing;
			plotargs...)
end
  ╠═╡ =#

# ╔═╡ ea49493a-9061-4c2e-95f4-a554cd46b616
md"""
## Importing the Shield:
"""

# ╔═╡ 25ce53a0-eebd-4fd6-9d26-3101b7789cb5
#=╠═╡
shield = robust_grid_deserialization(shield_file["data"] |> IOBuffer)
  ╠═╡ =#

# ╔═╡ 57c21c05-6563-4456-a29b-ef024d533eef
#=╠═╡
safe_actions(state) = int_to_actions(CCAction, get_value(box(shield, state)))
  ╠═╡ =#

# ╔═╡ f1d93191-b15b-4a9d-901b-8ba3c9950f22
function short_action_description(x) 
	actions = sort(int_to_actions(CCAction, x))
	action_dict = Dict(backwards => 'b', neutral => 'n', forwards => 'f')
	join([action_dict[a] for a in actions], ", ")
end

# ╔═╡ dd8cd700-25bb-447b-9108-e7504c95fe10
#=╠═╡
function draw_shield(shield::Grid, actions; v_ego=0, plotargs...)
	
	square = box(shield, [v_ego, 0, 0])
	index = square.indices[1]
	slice = [index, :, :]

	labels = [x for x in 0:length(ccshieldcolors)-1]
	labels = sort(labels, by=l -> length(int_to_actions(CCAction, l)))
	labels = [short_action_description(x) for x in labels]
	
	
	draw(shield, slice, 
		colors=ccshieldcolors,
		color_labels=labels,
		legend=:outerright,
		size=(800, 600),
		xlabel="v_front",
		ylabel="distance",
		title="v_ego=$v_ego";
		plotargs...)
end
  ╠═╡ =#

# ╔═╡ 1b55a282-6ece-4d54-862d-b7d792af7233
#=╠═╡
draw_shield(shield, CCAction, v_ego=v_ego)
  ╠═╡ =#

# ╔═╡ 8df19300-0705-4fc8-8101-d123f8e2e223
md"""
### Shielding the strategy

Behaviour when the shield intervenes can happen in one of two ways, controlled by the argument `deterministic`. 

If the argument is `true`, the policy will choose an action even if its preferred action is blocked by the shield. Instead, it will choose the best outcome among the allowed actions.

If the argument is `false`, no choice will be made in the case where the policy picks an unsafe action. Instead, the full list of safe actions is returned.
"""

# ╔═╡ d2e4e35c-035c-4a9b-b81d-61a9712f39eb
function get_action(shield::Grid, regressors, action_map, state,
	deterministic)
	
	policy_choice = get_action(regressors, action_map, state)
	
	if state ∉ shield
		return policy_choice
	end

	square = box(shield, state)
	allowed = int_to_actions(CCAction, get_value(square))

	if policy_choice[1] ∈ allowed
		return policy_choice
	elseif deterministic
		if length(allowed) == 0
			allowed = instances(CCAction)
		end
		return get_action(regressors, action_map, state; allowed)
	else
		return allowed
	end
end

# ╔═╡ 1a27105e-58d5-4dc2-bb7e-f7392a39c900
begin
	function get_strategy(file)
		jsondict = file |> JSON.parse
		regressors = jsondict["regressors"]["(1)"]["regressor"]
		action_map = get_action_map(jsondict)
		state -> get_action(regressors, action_map, state)
	end
	
	function get_strategy(file::Dict)
		get_strategy(file["data"] |> IOBuffer)
	end
	
	function get_strategy(file_path::AbstractString)
		open(file_path, "r") do file
			get_strategy(file)
		end
	end
end

# ╔═╡ 95a3046c-cfba-41f0-92ee-bd346ed63727
#=╠═╡
strategy = get_strategy(selected_file)
  ╠═╡ =#

# ╔═╡ 338a7dce-c680-421c-af11-716f6820e9b3
#=╠═╡
call() do
	draw(strategy, -9, 20, 0, 200, 0.5, [v_ego, :, :],
		xlabel="v_front",
		ylabel="distance",
		title="Imported Strategy",
		size=(800,600))

	 scatter!([], [], markeralpha=0, label="v_ego=$v_ego")
end
  ╠═╡ =#

# ╔═╡ a0c15b19-0dbd-4e5a-aff2-574755444ce0
begin
	function get_shielded_strategy(strategy_file, shield_file, deterministic)
		
		jsondict = strategy_file |> JSON.parse
		regressors = jsondict["regressors"]["(1)"]["regressor"]
		action_map = get_action_map(jsondict)
	
		shield = robust_grid_deserialization(shield_file)
		
		state -> get_action(shield, regressors, action_map, state, 
			deterministic)
	end
	
	function get_shielded_strategy(strategy_file::Dict, shield_file::Dict, 
			deterministic)
		
		get_shielded_strategy(
			strategy_file["data"] |> IOBuffer,
			shield_file["data"] |> IOBuffer,
			deterministic)
	end
	
	function get_shielded_strategy(strategy_path::AbstractString, 
			shield_path::AbstractString,
			deterministic)
		
		open(strategy_path, "r") do strategy_file
			open(shield_path, "r") do shield_file
				get_shielded_strategy(strategy_file, shield_file, deterministic)
			end
		end
	end
end

# ╔═╡ 3e77c1a7-6337-4c81-a8b1-e4a1da4e9595
#=╠═╡
# Deterministic strategy returns only one action for each state: The safe action which has the lowest Q-value

deterministic_shielded_strategy = get_shielded_strategy(selected_file, shield_file, true)
  ╠═╡ =#

# ╔═╡ ec36b03b-87e8-45e3-a72e-e6598c82735c
#=╠═╡
actions_to_int(CCAction, deterministic_shielded_strategy((v_ego, v_front, distance)))
  ╠═╡ =#

# ╔═╡ 98e22a5e-4745-44ba-81ea-a5372bd8e5ea
#=╠═╡
# Nondeterministic strategy may return multiple actions. If the action with the lowest Q-value overall is safe, that one is returned. Otherwise, it returns all allowed actions.

nondeterministic_shielded_strategy = get_shielded_strategy(selected_file, shield_file, false)
  ╠═╡ =#

# ╔═╡ c276f624-eae7-4da1-acab-fd27290e009a
#=╠═╡
shielded_strategy = get_shielded_strategy(selected_file, shield_file, false)
  ╠═╡ =#

# ╔═╡ 50c5a555-3d29-4714-8a5e-9e2469a60ab3
#=╠═╡
shielded_strategy((3, 3, 41))
  ╠═╡ =#

# ╔═╡ 5d113f75-b531-445a-bcfb-a2ba114e9b5d
#=╠═╡
shielded_strategy((10, 4, 81))
  ╠═╡ =#

# ╔═╡ 905751ca-cda2-485a-9e23-703e1706b6aa
#=╠═╡
call() do
	draw(shielded_strategy, -9, 20, 0, 200, 0.2, [v_ego, :, :],
		colors=strategycolors,
		actions=instances(CCAction),
		xlabel="v_front",
		ylabel="distance",
		title="Shielded Strategy",
		size=(800,600))

	 scatter!([], [], markeralpha=0, label="v_ego=$v_ego")
end
  ╠═╡ =#

# ╔═╡ f176f029-dd07-4fdd-bdd5-d4491e5382a3
function get_shielded_strategy_int(policy_file, shield_file, deterministic)
	
	shielded_strategy = get_shielded_strategy(policy_file, shield_file, deterministic)
	
	policy(state) = actions_to_int(CCAction, shielded_strategy(state))
end

# ╔═╡ 73167c7e-9203-4cc3-93d5-db88854ff760
#=╠═╡
shielded_strategy_int = get_shielded_strategy_int(selected_file, shield_file, true)
  ╠═╡ =#

# ╔═╡ 4e726cca-1569-44d1-860d-4cc618a9950f
#=╠═╡
get_shielded_strategy_int(selected_file, shield_file, false)((3, 3, 41))
  ╠═╡ =#

# ╔═╡ 1f52f4a4-6bcb-4e63-91a5-bd04ee9aa94c
# ╠═╡ skip_as_script = true
#=╠═╡
get_shielded_strategy(
	"path/to/CC-shielded.json",
	"path/to/CCShields/old testshield CC 192 samples with G of 0.5.shield",
	false
)([10, 10, 100])
  ╠═╡ =#

# ╔═╡ 789cf5c3-12b1-4902-b802-97fe60718972
md"""
### Detecting Interventions

Maybe this should be done in the `get_action` function, but I've coded myself into a corner here.
"""

# ╔═╡ 9962612b-1318-4c78-b051-27e64fa467cb
function intervention_occurred(shield, regressors, action_map, state)
	if state ∉ shield
		return false
	end
	
	allowed = int_to_actions(CCAction, get_value(box(shield, state)))
	
	if length(allowed) == 0
		return false
	end
	
	return get_action(regressors, action_map, state)[1] ∉ allowed
end

# ╔═╡ dd2f447b-ac67-47a8-ac34-4b43a28336b1
begin
	function get_intervention_checker(strategy_file, shield_file)
		jsondict = strategy_file |> JSON.parse
		regressors = jsondict["regressors"]["(1)"]["regressor"]
		action_map = get_action_map(jsondict)
	
		shield = robust_grid_deserialization(shield_file)
		
		state -> intervention_occurred(shield, regressors, action_map, state)
	end
	
	function get_intervention_checker(strategy_file::Dict, shield_file::Dict)
		get_intervention_checker(
			strategy_file["data"] |> IOBuffer,
			shield_file["data"] |> IOBuffer)
	end
	
	function get_intervention_checker(strategy_path::AbstractString, 
			shield_path::AbstractString)
		
		open(strategy_path, "r") do strategy_file
			open(shield_path, "r") do shield_file
				get_intervention_checker(strategy_file, shield_file)
			end
		end
	end
end

# ╔═╡ c249546e-5968-4034-a1af-4356d10a924c
#=╠═╡
intervened = get_intervention_checker(selected_file, shield_file)
  ╠═╡ =#

# ╔═╡ 4f6648c3-d119-4f31-8171-7ad186a3bbfe
#=╠═╡
intervened((3, 3, 3))
  ╠═╡ =#

# ╔═╡ c6e0e33f-ad0b-4f0b-8240-c229e044bbaa
#=╠═╡
intervened((3, 5, 30))
  ╠═╡ =#

# ╔═╡ 1fb20e9c-0d72-46d4-9f83-7b5b138a7b39
#=╠═╡
intervened((10, 4, 78))
  ╠═╡ =#

# ╔═╡ 29a9d1e2-b0d0-46f2-bbff-ba30f13d3ac0
md"""
## Testing the strategies

Get policy from selected file, then apply it to a state. Then, the policy is evaluated. The expected loss should match the UPPAAL model.
"""

# ╔═╡ 40c51d4c-d764-4d75-869d-0b105acc2d9e
#=╠═╡
s = [v_ego, v_front, distance]
  ╠═╡ =#

# ╔═╡ 885642a9-1043-4ed8-95fd-f8af3f50afad
#=╠═╡
shielded_strategy(s)
  ╠═╡ =#

# ╔═╡ ea2222ac-262c-4273-a2e8-4b40a01ab207
#=╠═╡
strategy(s)
  ╠═╡ =#

# ╔═╡ 9c9c4cd2-50fe-4ff1-b6d6-1ee8ad0e71a5
#=╠═╡
safe_actions(s)
  ╠═╡ =#

# ╔═╡ 28026526-6aed-4719-aaf9-2c929e1e7bb4
#=╠═╡
intervened(s)
  ╠═╡ =#

# ╔═╡ d38c8f61-0bb5-47e0-b7e5-f3dedb4e0dbe
#=╠═╡
get_predicted_outcome(regressor, s)
  ╠═╡ =#

# ╔═╡ e9d84aba-2b52-4b28-a1ba-c4887bb07a07
#=╠═╡
get_action(regressors, action_map, s)
  ╠═╡ =#

# ╔═╡ 4fe593dc-46e8-4bac-bffc-ccde586c26ae
#=╠═╡
get_shielded_strategy(selected_file, shield_file, false)(s)
  ╠═╡ =#

# ╔═╡ e92bdacc-c0f6-446b-9e05-57f9bbafdec8
#=╠═╡
get_action(shield, regressors, action_map, s, false)
  ╠═╡ =#

# ╔═╡ 45578a95-4616-4b1a-a883-7b0ace0b663e
#=╠═╡
get_action(shield, regressors, action_map, s, true)
  ╠═╡ =#

# ╔═╡ 3e4bfc91-1cdf-4781-994c-5d4d68a58c59
#=╠═╡
intervened(s)
  ╠═╡ =#

# ╔═╡ 030cd8cb-d3a5-4491-8a70-35eaeb1de581
#=╠═╡
strategy(s)
  ╠═╡ =#

# ╔═╡ e2fcf89d-ac98-4c3f-850c-0678892f5986
#=╠═╡
shielded_strategy(s)
  ╠═╡ =#

# ╔═╡ f103414d-3285-4a30-bc24-80fe64b50aea
#=╠═╡
shielded_strategy_int(s)
  ╠═╡ =#

# ╔═╡ c73b32ef-0ab2-4721-b4eb-25b3bcbb1864
#=╠═╡
int_to_actions(CCAction, get_value(box(shield, s)))
  ╠═╡ =#

# ╔═╡ 74bf1b3f-d07a-42ec-a179-dd061a479ac9
function loss(crash_cost, states)
	D::Float64 = 0
	crash = false
	for s in states
		D += s[3]
		crash = crash || s[3] <= 0
	end

	if crash
		return D/1000 + crash_cost
	else
		return D/1000
	end
end

# ╔═╡ de381f10-0d30-4eba-a342-06052232386d
function detect_crash(states)
	D = 0
	crash = false
	for s in states
		crash = crash || s[3] <= 0
	end

	crash
end

# ╔═╡ e678e366-c761-413f-9a18-e11ac94e57bf
function crashes(mechanics, policy; duration=120, runs=1000)
	crashes::Float64 = 0

	# The policy function should return a set of allowed actions
	unwrapped_policy(s) = begin
		result = policy(s)
		if length(result) != 1
			throw("To evaluate a policy, it must return exactly one allowed action. However, this policy returned $(result)")
		end
		result[1]
	end
	
	for _ in 1:runs
		s0 = (0., 0., 10.)
		front_behaviour = get_random_front_behaviour(mechanics)
		
		states, _ = simulate_sequence(mechanics, 
				front_behaviour, 
				duration, 
				s0, 
				unwrapped_policy)

		if detect_crash(states)
			crashes += 1
		end
	end
	
	(;crashes, runs)
end

# ╔═╡ 27187a02-c86c-40a3-9e5e-d7e8440b3511
function evaluate(mechanics, crash_cost, policy; duration=120, runs=1000)
	# The policy function should return a set of allowed actions
	unwrapped_policy(s) = begin
		result = policy(s)
		if length(result) != 1
			throw("To evaluate a policy, it must return exactly one allowed action. However, this policy returned $(result)")
		end
		result[1]
	end
	
	accumulator::Float64 = 0
	
	for _ in 1:runs
		s0 = (0., 0., 10.)
		front_behaviour = get_random_front_behaviour(mechanics)
		
		states, _ = simulate_sequence(mechanics, 
				front_behaviour, 
				duration, s0, 
				unwrapped_policy)
		
		accumulator += loss(crash_cost, states)
	end
	
	accumulator/runs
end

# ╔═╡ 93645491-974f-4975-8ee6-9c193ba46b6e
#=╠═╡
evaluate(ccmechanics, 0, strategy, runs=1000)
  ╠═╡ =#

# ╔═╡ c681b0be-bed1-4625-aac3-dacf4f58dd04
#=╠═╡
crashes(ccmechanics, strategy, runs=1000)
  ╠═╡ =#

# ╔═╡ b584d946-9b11-4fab-b6b4-4d255ecee520
#=╠═╡
evaluate(ccmechanics, 0, deterministic_shielded_strategy, runs=1000)
  ╠═╡ =#

# ╔═╡ 69dfd816-6054-434d-a08f-9ea82d5a5e7b
#=╠═╡
crashes(ccmechanics, deterministic_shielded_strategy, runs=1000)
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
Colors = "~0.12.8"
JSON = "~0.21.3"
Plots = "~1.35.5"
PlutoUI = "~0.7.48"
StatsBase = "~0.33.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "32f8df8dfed94b0885775a029335517efd6acad7"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "84259bb6172806304b9101094a7cc4bc6f56dbc6"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "5856d3031cdb1f3b2b6340dfdc66b6d9a149a374"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.2.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "00a9d4abadc05b9476e937a5557fcce476b9e547"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.69.5"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "bc9f7725571ddb4ab2c4bc74fa397c1c5ad08943"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.69.1+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "a97d47758e933cd5fe5ea181d178936a9fc60427"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.5.1"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "ab9aa169d2160129beb241cb2750ca499b4e90e9"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.17"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "6872f9594ff273da6d13c7c1a1545d5a8c7d0c1c"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.6"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "3c3c4a401d267b04942545b1e964a20279587fd7"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "3d5bf43e3e8b412656404ed9466f1dcbf7c50269"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.0"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "21303256d239f6b484977314674aef4bb1fe4420"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "0a56829d264eb1bc910cf7c39ac008b5bcb5a0d9"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.35.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "efc140104e6d0ae3e7e30d56c98c4a927154d684"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.48"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "d12e612bba40d189cead6ff857ddb67bd2e6a387"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "9b1c0c8e9188950e66fc28f40bfe0f8aac311fe0"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.7"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╠═f474fbb6-900b-4e18-8ba0-12a85358d077
# ╠═bdb61d68-2ac9-11ed-1b20-9dc154fc45d0
# ╠═12556d17-f66f-4db3-8718-3576b8a2c8dc
# ╠═c2d7aa5e-70b3-4c9c-8196-7f07959e9802
# ╟─5660755c-7418-4ffd-9e57-0f1a5d168fef
# ╠═3e77c1a7-6337-4c81-a8b1-e4a1da4e9595
# ╠═98e22a5e-4745-44ba-81ea-a5372bd8e5ea
# ╠═ec36b03b-87e8-45e3-a72e-e6598c82735c
# ╟─93288f99-4efd-46c8-ab2b-93432a4ddd06
# ╠═2dde1bfc-f79d-4936-9a82-6006c550e2ee
# ╟─c41edb3e-7051-4ea6-a1cd-636f89fe9483
# ╟─c870988d-96aa-4114-9901-35472f341d16
# ╟─eeebe049-8b0b-4e3e-8cb2-4ee89e241273
# ╟─2abe9b5e-8673-4627-bca0-5d2b4940eb27
# ╟─fd0135aa-78c8-49cb-9ee5-dc1599881d16
# ╠═57c21c05-6563-4456-a29b-ef024d533eef
# ╠═c276f624-eae7-4da1-acab-fd27290e009a
# ╠═73167c7e-9203-4cc3-93d5-db88854ff760
# ╠═95a3046c-cfba-41f0-92ee-bd346ed63727
# ╠═885642a9-1043-4ed8-95fd-f8af3f50afad
# ╠═ea2222ac-262c-4273-a2e8-4b40a01ab207
# ╠═9c9c4cd2-50fe-4ff1-b6d6-1ee8ad0e71a5
# ╠═28026526-6aed-4719-aaf9-2c929e1e7bb4
# ╠═c249546e-5968-4034-a1af-4356d10a924c
# ╠═4f6648c3-d119-4f31-8171-7ad186a3bbfe
# ╠═c6e0e33f-ad0b-4f0b-8240-c229e044bbaa
# ╠═1fb20e9c-0d72-46d4-9f83-7b5b138a7b39
# ╠═50c5a555-3d29-4714-8a5e-9e2469a60ab3
# ╠═5d113f75-b531-445a-bcfb-a2ba114e9b5d
# ╟─8894e942-b6f6-4893-bfea-ebaa2f9c58f0
# ╠═9a3af469-7dc0-4d29-a375-bdc79415e950
# ╠═3eb461a0-6f98-43a7-986a-423afb55766c
# ╠═667f2a9f-633e-43d4-9421-dd31338995f2
# ╟─b3e2a24a-fab6-414e-be8b-351a0e7d1b1c
# ╟─35e3c5a6-39e1-4f92-94af-09ff6148b358
# ╠═4d616f3f-3bb9-4a14-b785-838786849a4d
# ╠═293c4fa8-9fe2-4fd7-ad4f-bbdcc2e0614b
# ╟─bef24a24-6229-497f-8d55-2b51478289e6
# ╠═0da72168-bc4d-41e0-865a-a29ddd762ddf
# ╟─5204d97e-6e04-4525-818b-a7ed642aec0d
# ╠═d38c8f61-0bb5-47e0-b7e5-f3dedb4e0dbe
# ╠═1a27105e-58d5-4dc2-bb7e-f7392a39c900
# ╠═2cdfba1e-b6f2-4e9f-82fd-da14ef7415f5
# ╠═7f840af3-bff8-4548-9d31-cf3c64fa9faf
# ╠═e9d84aba-2b52-4b28-a1ba-c4887bb07a07
# ╠═f969579d-e392-4dce-88ba-6a5da83b599f
# ╟─338a7dce-c680-421c-af11-716f6820e9b3
# ╟─ea49493a-9061-4c2e-95f4-a554cd46b616
# ╠═25ce53a0-eebd-4fd6-9d26-3101b7789cb5
# ╠═1b55a282-6ece-4d54-862d-b7d792af7233
# ╟─dd8cd700-25bb-447b-9108-e7504c95fe10
# ╠═f1d93191-b15b-4a9d-901b-8ba3c9950f22
# ╟─8df19300-0705-4fc8-8101-d123f8e2e223
# ╟─a0c15b19-0dbd-4e5a-aff2-574755444ce0
# ╠═905751ca-cda2-485a-9e23-703e1706b6aa
# ╠═4fe593dc-46e8-4bac-bffc-ccde586c26ae
# ╟─f176f029-dd07-4fdd-bdd5-d4491e5382a3
# ╠═4e726cca-1569-44d1-860d-4cc618a9950f
# ╠═d2e4e35c-035c-4a9b-b81d-61a9712f39eb
# ╠═e92bdacc-c0f6-446b-9e05-57f9bbafdec8
# ╠═45578a95-4616-4b1a-a883-7b0ace0b663e
# ╠═1f52f4a4-6bcb-4e63-91a5-bd04ee9aa94c
# ╟─789cf5c3-12b1-4902-b802-97fe60718972
# ╠═dd2f447b-ac67-47a8-ac34-4b43a28336b1
# ╠═9962612b-1318-4c78-b051-27e64fa467cb
# ╠═3e4bfc91-1cdf-4781-994c-5d4d68a58c59
# ╟─29a9d1e2-b0d0-46f2-bbff-ba30f13d3ac0
# ╠═40c51d4c-d764-4d75-869d-0b105acc2d9e
# ╠═030cd8cb-d3a5-4491-8a70-35eaeb1de581
# ╠═e2fcf89d-ac98-4c3f-850c-0678892f5986
# ╠═f103414d-3285-4a30-bc24-80fe64b50aea
# ╠═c73b32ef-0ab2-4721-b4eb-25b3bcbb1864
# ╟─74bf1b3f-d07a-42ec-a179-dd061a479ac9
# ╟─de381f10-0d30-4eba-a342-06052232386d
# ╟─e678e366-c761-413f-9a18-e11ac94e57bf
# ╠═27187a02-c86c-40a3-9e5e-d7e8440b3511
# ╠═93645491-974f-4975-8ee6-9c193ba46b6e
# ╠═c681b0be-bed1-4625-aac3-dacf4f58dd04
# ╠═b584d946-9b11-4fab-b6b4-4d255ecee520
# ╠═69dfd816-6054-434d-a08f-9ea82d5a5e7b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
