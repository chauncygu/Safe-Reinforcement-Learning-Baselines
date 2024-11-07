### A Pluto.jl notebook ###
# v0.19.32

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

# ╔═╡ bb902940-a858-11ed-2f11-1d6f5af61e4a
begin
	using Pkg
	Pkg.activate("..")
	using GridShielding
	using Plots
	using PlutoLinks
	using PlutoUI
	using Unzip
	using Printf
	using StatsBase
	#include("../Shared Code/PlotsDefaults.jl")
	include("../Shared Code/OilPump.jl")
	include("../Shared Code/OPShielding.jl")
	include("../Shared Code/FlatUI.jl")
	TableOfContents()
end

# ╔═╡ 6f5584c1-ea5e-49ee-afc0-25abde4e295a
md"""
# Shielding the Oil Pump Control Problem

From [**A “Hybrid” Approach for Synthesizing Optimal
Controllers of Hybrid Systems: A Case Study of
the Oil Pump Industrial Example**](https://www.researchgate.net/publication/221960725_A_Hybrid_Approach_for_Synthesizing_Optimal_Controllers_of_HybridSystems_A_Case_Study_of_the_Oil_Pump_Industrial_Example)


The oil pump example was a real industrial case provided by the German company HYDAC ELECTRONICS GMBH, and studied at length within the European research project Quasimodo. The whole system, depicted by Fig. 1, consists of a machine, an accumulator, a reservoir and a pump. The machine consumes oil periodically out of the accumulator with a duration of $20 s$ (second) for one consumption cycle. The profile of consumption rate is shown in Fig. 2. The pump adds oil from the reservoir into the accumulator with power $2.2 l/s$ (liter/second).

![Left: The oil pump system. (This picture is based on [3].) Right:Consumption rate of the machine in one cycle.](https://i.imgur.com/l0ecK8u.png)

Control objectives for this system are: by switching on/off the pump at cer-
tain time points ensure that

- Safety, $R_s$: the system can run arbitrarily long while maintaining v(t) within $[V_{min} , V_{max} ]$ for any time point t, where v(t) denotes the oil volume in the accumulator at time $t$, $V_{min} = 4.9 l$ (liter) and $V_{max} = 25.1 l$ ; 

and considering the energy cost and wear of the system, a second objective:

- Optimality, $R_o$: minimize the average accumulated oil volume in the limit, i.e. minimize

$\lim_{T \to \infty} {1 \over T} \int_{t=0}^{T} v(t) \,\text dt$

Both objectives should be achieved under two additional constraints:

- Pump latency, $R_{pl}$: there must be a latency of at least $2 s$ between any two consecutive operations of the pump; and
- Robustness, $R_r$: uncertainty of the system should be taken into account:
   - fluctuation of consumption rate (if it is not $0$), up to $f = 0.1 l/s$
   - imprecision in the measurement of oil volume, up to $\epsilon = 0.06 l$ ;
"""

# ╔═╡ e4f088b7-b48a-4c6f-aa36-fc9fd4746d9b
md"""
## Preface
"""

# ╔═╡ 5ae3173f-6abb-4f38-94f8-90300c93d0e9
call(f) = f()

# ╔═╡ 35fbdec7-b673-40a9-8e49-2e19c596b71b
md"""
## Mechanics and Gridification
"""

# ╔═╡ 67d83ab6-8d99-4067-aafc-dee1026eb1dc
m = OPMechanics()

# ╔═╡ 3e447971-62d4-4d34-95de-c6dcfe1a281f
md"""
The state variable is going to be: $(t, v, p, l)$ where

 - ⁣$t$ is the time in the consumption cycle
 - ⁣$v$ is the volume of oil in the tank
 - ⁣$p$ is the pump status (Corresponding to the automaton locations *on* and *off*.)
 - ⁣$l$ is the latency-timer controlling how often the pump can switch state
"""

# ╔═╡ 1687a47c-c3f6-4518-ac46-e97b240ad323
md"""
Note that the variable $p$ represents an automaton location and can therefore only assume the values $0$ and $1$. For this reason, the granularity is fixed to $1$.

`granularity_t =` $(@bind granularity_t NumberField(0.001:0.001:4, default=1))

`granularity_v =` $(@bind granularity_v NumberField(0.001:0.001:4, default=1))

`granularity_p = 1.0`

`granularity_l =` $(@bind granularity_l NumberField(0.001:0.001:m.time_step, default=1))

"""

# ╔═╡ 1c3a6140-cd65-4081-99f3-397b74e6bf89
granularity = [granularity_t, granularity_v, 1, granularity_l]

# ╔═╡ ae621a99-56e3-4a93-8af0-096c3a6f00f0
grid = get_op_grid(m, granularity)

# ╔═╡ be055e02-7ef6-4a63-8c95-d6c2bfdc799a
md"""
Total partitions: **$(length(grid.array))**
"""

# ╔═╡ cfe8387f-a127-4e46-88a6-40d9442fe4b1
md"""
## Simulation Model
"""

# ╔═╡ d2300c36-906c-4351-952a-3a5176338649
randomness_space = get_randomness_space(m)

# ╔═╡ ce2ecf63-c2dc-4c6b-9a60-1a934e915ba2
md"""

Select number of samples per axis. 

The location variable $p$ can only ever have 1 sample per axis, since it can only take on values 0 and 1.

$(@bind samples_per_axis_selected NumberField(1:30, default=2))
"""

# ╔═╡ f998df54-b9d8-4ab5-85c4-8266c4e7a01c
samples_per_axis = [samples_per_axis_selected, samples_per_axis_selected, 1, samples_per_axis_selected]

# ╔═╡ 04f6c10f-ee06-40f3-969d-9197504c9f61
simulation_function = get_simulation_function(m)

# ╔═╡ 90efd733-ea84-46c4-80a5-556f23dc4192
simulation_model = SimulationModel(simulation_function, randomness_space, samples_per_axis)

# ╔═╡ f81f53ce-d81e-429d-ac80-a3edd2f76eac
md"""
## Some Debug Stuff
"""

# ╔═╡ 9f1c8c62-0ad1-4c93-8f0a-ef0f61f2bcf4
md"""
## Importing Safe Strategy

Optionally, you can import an existing safe strategy.
"""

# ╔═╡ 5ff2a592-e30e-43ad-938e-5396f94f713e
md"""
**Pick your shield:** 

`selected_file` = $(@bind selected_file PlutoUI.FilePicker([MIME("application/octet-stream")]))
"""

# ╔═╡ f589d4eb-5f83-449b-8271-56fc1b008b83
imported_shield = 
	if selected_file !== nothing
		robust_grid_deserialization(selected_file["data"] |> IOBuffer)
	else
		nothing
	end

# ╔═╡ 252ac6f4-ae88-4647-91cc-7f29e0a1a015
md"""
## Synthesising Safe Strategy
"""

# ╔═╡ 40116c98-2afb-48d8-a7d6-de03c5bc119c
grid, simulation_model; @bind make_shield_button CounterButton("Make Shield")

# ╔═╡ f65de4dd-438b-43c2-9571-adc3fa03fb09
reachability_function = get_barbaric_reachability_function(simulation_model)

# ╔═╡ d57587e2-a9f3-4e10-9679-325f716882e9
if make_shield_button > 0
	reachability_function_precomputed = 
		get_transitions(reachability_function, PumpStatus, grid)
end

# ╔═╡ 084c26b7-2786-4aea-af07-43e6adee06cf
@bind max_steps NumberField(0:1000, default=10)

# ╔═╡ 09496aef-95be-43b8-95d1-5cdaa9da50b9
if make_shield_button > 0 && imported_shield === nothing

	## The actual call to make_shield ##
	
	shield, max_steps_reached = 
		make_shield(reachability_function_precomputed, PumpStatus, grid; max_steps)

	
elseif imported_shield === nothing
	shield, max_steps_reached = grid, true
else
	shield, max_steps_reached = imported_shield, false
end

# ╔═╡ 56781a46-51a5-425d-aea8-bcfd4820da88
if max_steps_reached
md"""
!!! danger "NB"
	Synthesis not complete. 

	Either synthesis hasn't started, or `max_steps` needs to be increased to obtain an infinite-horizon strategy.
"""
end

# ╔═╡ 671e75ef-7c4d-4fc5-a0a3-66d0f59e4778
md"""
Show barbaric transition $(@bind show_tv CheckBox()) 
"""

# ╔═╡ 1e2dcb19-8e61-45b9-a033-0e28406b1511
md"""
Select which axes to display. Order is ignored.

$(@bind index_1 Select(
	[1 => "t",
	2 => "v",
	3 => "p",
	4 => "l",]
))
$(@bind index_2 Select(
	[1 => "t",
	2 => "v",
	3 => "p",
	4 => "l",],
	default=2
))
"""

# ╔═╡ bf83ba44-8900-48c8-a172-161337181e41
begin
	xlabel = min(index_1, index_2)
	ylabel = max(index_1, index_2)
	state_variables = Dict(1 => "t", 2 => "v", 3 => "p", 4 => "l")
	xlabel = state_variables[xlabel]
	ylabel = state_variables[ylabel];
end

# ╔═╡ 19e5297d-c095-4e72-a2b1-e63d682d1587


# ╔═╡ 3131d5d5-34ef-4440-ae19-198b6ae8d54a
grid, simulation_model; @bind make_animation_button CounterButton("Make Animation")

# ╔═╡ 7692cddf-6b37-4be2-847f-afb6d34e44ab
md"""
### Select State to preview

!!! info "Tip"
	This cell affects multiple other cells across the notebook. Drag it around to make interaction easier.

`t =` 
$(@bind t NumberField(0:shield.granularity[1]:20-shield.granularity[1]))

`v =` 
$(@bind v NumberField(m.v_min:shield.granularity[2]:m.v_max))

`p =`
$(@bind p Select([Int(a) for a in instances(PumpStatus)]))

`l =`
$(@bind l NumberField(-shield.granularity[4]:shield.granularity[4]:shield.bounds.upper[4]-shield.granularity[4]))

`action =` 
$(@bind action Select(instances(PumpStatus) |> collect))

"""

# ╔═╡ 8751340a-f41a-46fa-8f6d-cc9ca132e260
partition = box(shield, (t, v, p, l))

# ╔═╡ 5d35a493-0195-46f7-bdf6-013fde056a1e
sample_count = (length(SupportingPoints(samples_per_axis, partition)))

# ╔═╡ 7749a8ca-2f38-41c7-9372-df06ce54b919
Bounds(partition)

# ╔═╡ c6aec984-3963-41f3-9281-e267d1c8ac78
supporting_points = SupportingPoints(samples_per_axis, partition)

# ╔═╡ 31699662-ddfc-45a8-b963-f0b03b7c71c2
supporting_points |> collect

# ╔═╡ d099b12b-9e8e-482f-82ed-a4681a424d2e
slice = let
	slice = Any[i for i in partition.indices]
	slice[index_1] = Colon()
	slice[index_2] = Colon()
	slice
end

# ╔═╡ 5d0400fb-2ad5-4d07-9115-49024913c3fa
if make_animation_button > 0
	shield2 = shield
	a = @animate for i in 1:10
		shield2 = shield_step(reachability_function_precomputed, PumpStatus, shield2)
		
		draw(something(shield2, grid), slice,
			legend=:outerright, 
			colors=opshieldcolors,
			color_labels=opshieldlabels;
			xlabel, ylabel)
	end
end

# ╔═╡ 47fdca20-194d-4535-a4a3-25a7509324b5
make_animation_button > 0 && 
	gif(a, fps=15)

# ╔═╡ b83a55ee-f2a3-4b0b-8e0a-12dc6caf5075
possible_outcomes(simulation_model, partition, action)

# ╔═╡ cd732487-5c1b-487e-b037-4523a7389365
[Partition(grid, i) |> Bounds 
	for i in reachability_function(partition, action)]

# ╔═╡ caf38611-4ba4-4b10-99aa-d72252a8b60d
let
	draw(something(shield, grid), slice,
		legend=:outerright, 
		colors=opshieldcolors,
		color_labels=opshieldlabels;
		xlabel, ylabel)
	
	if show_tv
		draw_barbaric_transition!(simulation_model, partition, action, slice)
	end
	#==
	plot!([], 
		line=(3, colors.PETER_RIVER),
		label="consumption ")
	 plot!(twinx(), t -> consumption_rate(abs(t%m.period)), 
		line=(3, colors.PETER_RIVER),
		ylim=(0,3),
		label=nothing) 
	==#
	plot!()
end

# ╔═╡ 0a0285ca-a922-467f-ae91-b7a96ee43588
let
	# For the paper
	slice = deepcopy(slice)
	slice[3] = 1 # Pump status off
	draw(something(shield, grid), slice,
		legend=nothing, 
		colors=opshieldcolors,
		color_labels=opshieldlabels;
		title="When \$pump~status=off\$",
		titlefontsize=10,
		xlabel="Time (s)", ylabel="Volume (l)")
	
	if show_tv
		draw_barbaric_transition!(simulation_model, partition, action, slice)
	end
	plot!([], 
		line=(3, colors.ASBESTOS),
		label="Consumption ")
	
	#==# plot!(twinx(), t -> consumption_rate(abs(t%m.period)), 
		line=(3, colorant"#717171"),
		ylim=(0,3),
		ylabel="Consumption (l/s)",
		label=nothing) #==#
	plot!(size=(halfpage, 240))
end

# ╔═╡ cacf2987-28ef-42a2-b6ff-26e512b84856
let
	# For the paper
	slice = deepcopy(slice)
	slice[3] = 2 # Pump status on
	draw(something(shield, grid), slice,
		legend=:topleft, 
		colors=opshieldcolors,
		color_labels=opshieldlabels;
		title="When \$pump~status=on\$",
		titlefontsize=10,
		xlabel="Time (s)", ylabel="Volume (l)")
	
	if show_tv
		draw_barbaric_transition!(simulation_model, partition, action, slice)
	end
	plot!([], 
		line=(3, colors.ASBESTOS),
		label="Consum.")
	
	#==# plot!(twinx(), t -> consumption_rate(abs(t%m.period)), 
		line=(3, colorant"#717171"),
		ylim=(0,3),
		ylabel="Consumption (l/s)",
		label=nothing) #==#
	plot!(size=(halfpage, 240))
end

# ╔═╡ b7d70793-92b5-4812-a6dd-7302ee9e6ae4
shield_is_valid(shield)

# ╔═╡ f6bab622-4b1d-41ec-ae54-61915fca3b2c
reachability_function′(partition, a) = begin
	result = reachability_function(partition, a)
	result = map(r -> Partition(partition.grid, r), result)
	result = map(r -> (opshieldcolors[get_value(r)+1], (Bounds(r))), result)
end

# ╔═╡ 7358338f-47d9-4cb1-a868-f89b0162e72d
reachability_function′(partition, off)

# ╔═╡ 258ec4cf-4193-4b14-bf4c-53f83eca96ae
reachability_function′(partition, on)

# ╔═╡ 4d169b72-54f8-4325-adec-f53d18e54fae
md"""
## Check Safety
"""

# ╔═╡ dae2fc1d-38d0-48e1-bddc-3b490648648b
md"""
Set the probability that the random agents selects the action "off" at any given time

`off_chance =` $(@bind off_chance NumberField(0:0.01:1, default=0.3))
"""

# ╔═╡ 4f01a075-b44b-467c-9f87-55df435b7bdd
random_agent(_...) = sample([on, off], [1 - off_chance, off_chance] |> Weights)

# ╔═╡ 794b75b0-5d14-4509-a293-0b657d6d0600
grid, simulation_model; @bind check_safety_button CounterButton("Check Safety")

# ╔═╡ a57c6670-6d88-4119-b5b1-7509a8806dae
shielded(something(shield, grid), (_...) -> action)((t, v, p, l))

# ╔═╡ 1b447b3e-0565-4dc5-b679-5102c946dec2
shielded_random_agent = shielded(shield, random_agent)

# ╔═╡ fdfa1b59-217e-4504-9d4f-2ad44c39cfd8
if check_safety_button > 0 let
	draw(shield, [:, :, 2, 1],
		colorbar=:right, 
		colors=opshieldcolors,
		color_labels=opshieldlabels)

	for _ in 1:10
		trace = 
			simulate_trace(m, (0., 10., Int(off), 0.), shielded(shield, shielded_random_agent), duration=19.9)
		
		plot!(trace.ts, trace.vs,
			line=(colors.WET_ASPHALT, 2),
			label=nothing)
	end
	plot!(xlabel="t", ylabel="v")
end end

# ╔═╡ dad2155a-562d-4340-afa3-cd6889ba216b
[1 2]*[1 1; 1 1]

# ╔═╡ aeba4953-dee5-4810-a3de-0fc191711e16
if check_safety_button > 0
	plot()
	for i in 1:10
		trace = 
			simulate_trace(m, (0., 10., Int(off), 0.), shielded_random_agent, duration=120)
		
		plot!(trace.elapsed, trace.vs,
			#line=(colors.WET_ASPHALT, 2),
			label="trace $i")
	end
	hline!([4.9, 25], label="safety constraints")
	plot!(xlabel="t", ylabel="v", legend=:topleft)
end

# ╔═╡ d77f23be-3a54-4c48-ab6d-b1c31adc3e25
if check_safety_button > 0
	unsafe, total, unsafe_trace = count_unsafe_traces(m, 
		shielded_random_agent, 
		run_duration=120, 
		runs=1000)
end

# ╔═╡ 150d8707-e8ef-4476-9378-9dd1c63036bf
if check_safety_button > 0 && unsafe > 0
Markdown.parse("""
!!! danger "Shield is Unsafe"
    There were $unsafe safety violations during the $total runs.
""")
elseif check_safety_button > 0
Markdown.parse("""
!!! success "Shield is Safe"
    There were no safety violations during the $total runs.
""")
end

# ╔═╡ ac138da0-fd64-4e35-ab26-e5803fa2d9b5
cost(m, shielded_random_agent)

# ╔═╡ e5a18013-48a1-4329-b238-65a606a82c9b
if check_safety_button > 0 && unsafe_trace !== nothing
	@bind state_index NumberField(1:length(unsafe_trace.actions), default=2)
end

# ╔═╡ f0a96c74-c73b-4763-992e-73d4aa542976
if check_safety_button > 0 && unsafe_trace != nothing let

	(;ts, vs, ps, ls, elapsed, actions) = unsafe_trace
	unsafe_trace′ = [ts, vs, ps, ls, elapsed, actions]

	t, v, p, l, a = ts[state_index], vs[state_index], ps[state_index], ls[state_index], actions[state_index]

	@info (;t, v, p, l, a)
	partition = box(shield, clamp_state(shield, (t, v, p, l)))
	
	slice = Any[i for i in partition.indices]
	slice[index_1] = Colon()
	slice[index_2] = Colon()
	slice

	draw(shield, slice,
		legend=:outerright, 
		colors=opshieldcolors,
		color_labels=opshieldlabels)
	xs, ys = unsafe_trace′[min(index_1, index_2)], unsafe_trace′[max(index_1, index_2)]
	x, y = xs[state_index], ys[state_index]
	
	plot!(xs, ys,
		xlims=(shield.bounds.lower[1], shield.bounds.upper[1]),
		ylims=(shield.bounds.lower[2], shield.bounds.upper[2]),
		line=(colors.WET_ASPHALT, 2);
		xlabel, ylabel)
	
	scatter!([x], [y],
		marker=(colors.WET_ASPHALT, 5, :+),
		msw=4)
end end

# ╔═╡ 4af0b349-5894-4da5-8c3b-9fbc466d94f5
if check_safety_button > 0 && unsafe_trace != nothing let 
	(;ts, vs, ps, ls, elapsed, actions) = unsafe_trace
	
	shielded_random_agent((ts[state_index], vs[state_index], ps[state_index], ls[state_index])),  actions[state_index - 1]
end end

# ╔═╡ 16598016-eb21-43da-a45b-bd09692125ca
call(() -> begin
	
	buff = IOBuffer(sizehint=length(shield.array))
	robust_grid_serialization(buff, shield)
	shield_description =  "samples $sample_count granularity $(grid.granularity).shield"
	
	md"""
	## Save
	Save as serialized julia object
	
	$(DownloadButton(buff.data, shield_description))
	"""
end)

# ╔═╡ 63b217ad-bb2c-420b-b327-2c9a28be0a90
let 
	buff = IOBuffer()
	
	println(buff, get_c_library_header(shield, "Samples used: $samples_per_axis"))
	
	md"""
	Dump into a C file (hard-coded into a `const char[]`)
	
	$(DownloadButton(buff.data, "shield_dump.c"))
	"""
end

# ╔═╡ Cell order:
# ╟─6f5584c1-ea5e-49ee-afc0-25abde4e295a
# ╟─e4f088b7-b48a-4c6f-aa36-fc9fd4746d9b
# ╠═bb902940-a858-11ed-2f11-1d6f5af61e4a
# ╠═5ae3173f-6abb-4f38-94f8-90300c93d0e9
# ╟─35fbdec7-b673-40a9-8e49-2e19c596b71b
# ╠═67d83ab6-8d99-4067-aafc-dee1026eb1dc
# ╟─3e447971-62d4-4d34-95de-c6dcfe1a281f
# ╟─1687a47c-c3f6-4518-ac46-e97b240ad323
# ╠═1c3a6140-cd65-4081-99f3-397b74e6bf89
# ╟─be055e02-7ef6-4a63-8c95-d6c2bfdc799a
# ╠═ae621a99-56e3-4a93-8af0-096c3a6f00f0
# ╟─cfe8387f-a127-4e46-88a6-40d9442fe4b1
# ╠═d2300c36-906c-4351-952a-3a5176338649
# ╟─ce2ecf63-c2dc-4c6b-9a60-1a934e915ba2
# ╠═f998df54-b9d8-4ab5-85c4-8266c4e7a01c
# ╠═5d35a493-0195-46f7-bdf6-013fde056a1e
# ╠═04f6c10f-ee06-40f3-969d-9197504c9f61
# ╠═90efd733-ea84-46c4-80a5-556f23dc4192
# ╟─f81f53ce-d81e-429d-ac80-a3edd2f76eac
# ╠═8751340a-f41a-46fa-8f6d-cc9ca132e260
# ╠═7749a8ca-2f38-41c7-9372-df06ce54b919
# ╠═c6aec984-3963-41f3-9281-e267d1c8ac78
# ╠═31699662-ddfc-45a8-b963-f0b03b7c71c2
# ╠═b83a55ee-f2a3-4b0b-8e0a-12dc6caf5075
# ╠═cd732487-5c1b-487e-b037-4523a7389365
# ╟─9f1c8c62-0ad1-4c93-8f0a-ef0f61f2bcf4
# ╟─5ff2a592-e30e-43ad-938e-5396f94f713e
# ╠═f589d4eb-5f83-449b-8271-56fc1b008b83
# ╟─252ac6f4-ae88-4647-91cc-7f29e0a1a015
# ╟─40116c98-2afb-48d8-a7d6-de03c5bc119c
# ╠═f65de4dd-438b-43c2-9571-adc3fa03fb09
# ╠═d57587e2-a9f3-4e10-9679-325f716882e9
# ╠═09496aef-95be-43b8-95d1-5cdaa9da50b9
# ╟─56781a46-51a5-425d-aea8-bcfd4820da88
# ╠═084c26b7-2786-4aea-af07-43e6adee06cf
# ╟─671e75ef-7c4d-4fc5-a0a3-66d0f59e4778
# ╟─1e2dcb19-8e61-45b9-a033-0e28406b1511
# ╠═d099b12b-9e8e-482f-82ed-a4681a424d2e
# ╠═bf83ba44-8900-48c8-a172-161337181e41
# ╠═19e5297d-c095-4e72-a2b1-e63d682d1587
# ╟─3131d5d5-34ef-4440-ae19-198b6ae8d54a
# ╠═5d0400fb-2ad5-4d07-9115-49024913c3fa
# ╠═47fdca20-194d-4535-a4a3-25a7509324b5
# ╟─caf38611-4ba4-4b10-99aa-d72252a8b60d
# ╠═0a0285ca-a922-467f-ae91-b7a96ee43588
# ╠═cacf2987-28ef-42a2-b6ff-26e512b84856
# ╟─7692cddf-6b37-4be2-847f-afb6d34e44ab
# ╠═b7d70793-92b5-4812-a6dd-7302ee9e6ae4
# ╠═f6bab622-4b1d-41ec-ae54-61915fca3b2c
# ╠═7358338f-47d9-4cb1-a868-f89b0162e72d
# ╠═258ec4cf-4193-4b14-bf4c-53f83eca96ae
# ╟─4d169b72-54f8-4325-adec-f53d18e54fae
# ╟─dae2fc1d-38d0-48e1-bddc-3b490648648b
# ╠═4f01a075-b44b-467c-9f87-55df435b7bdd
# ╟─794b75b0-5d14-4509-a293-0b657d6d0600
# ╠═a57c6670-6d88-4119-b5b1-7509a8806dae
# ╠═1b447b3e-0565-4dc5-b679-5102c946dec2
# ╟─fdfa1b59-217e-4504-9d4f-2ad44c39cfd8
# ╠═dad2155a-562d-4340-afa3-cd6889ba216b
# ╟─aeba4953-dee5-4810-a3de-0fc191711e16
# ╠═d77f23be-3a54-4c48-ab6d-b1c31adc3e25
# ╟─150d8707-e8ef-4476-9378-9dd1c63036bf
# ╠═ac138da0-fd64-4e35-ab26-e5803fa2d9b5
# ╠═e5a18013-48a1-4329-b238-65a606a82c9b
# ╟─f0a96c74-c73b-4763-992e-73d4aa542976
# ╠═4af0b349-5894-4da5-8c3b-9fbc466d94f5
# ╟─16598016-eb21-43da-a45b-bd09692125ca
# ╟─63b217ad-bb2c-420b-b327-2c9a28be0a90
