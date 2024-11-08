### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ ca24fa88-c330-11ed-16b7-655126a8c62d
begin
	using Plots
	include("FlatUI.jl");
end

# ╔═╡ 25fa7e5a-04f2-49df-a0b0-0a3f69d721c9
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	# Includes which are disabled in file
	using PlutoUI
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ 484bbefc-c519-4d8f-9f25-df8596deecd9
md"""
# Oil Pump Control Problem

From [**A “Hybrid” Approach for Synthesizing Optimal
Controllers of Hybrid Systems: A Case Study of
the Oil Pump Industrial Example**](https://www.researchgate.net/publication/221960725_A_Hybrid_Approach_for_Synthesizing_Optimal_Controllers_of_HybridSystems_A_Case_Study_of_the_Oil_Pump_Industrial_Example)


The oil pump example was a real industrial case provided by the German company HYDAC ELECTRONICS GMBH, and studied at length within the European research project Quasimodo. The whole system, depicted by Fig. 1, consists of a machine, an accumulator, a reservoir and a pump. The machine consumes oil periodically out of the accumulator with a duration of $20 s$ (second) for one consumption cycle. The profile of consumption rate is shown in Fig. 2. The pump adds oil from the reservoir into the accumulator with power $2.2 l/s$ (liter/second).

![Left: The oil pump system. (This picture is based on [3].) Right:Consumption rate of the machine in one cycle.](https://i.imgur.com/l0ecK8u.png)

Control objectives for this system are: by switching on/off the pump at certain time points ensure that

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

# ╔═╡ 7144e8f2-38af-4f32-8377-47841261bac5
md"""
# My Implementation

## Basics
"""

# ╔═╡ ebd20f9c-a927-4a50-98e4-2aec1f2c8e1b
# Wrapping all the constants which define the system into a neat little ball.
struct OPMechanics
	v_min::Float64
	v_max::Float64
	period::Float64
	time_step::Float64
	inflow::Float64
	fluctuation::Float64
	imprecision::Float64
	latency::Float64

	function OPMechanics(params...)
		new(params...)
	end

	function OPMechanics(;
			v_min=4.9,
			v_max=25.1,
			period=20,
			time_step=0.2,
			inflow=2.2,
			fluctuation=0.1,
			imprecision=0.06,
			latency=2)
		
		OPMechanics(v_min, v_max, period, time_step, inflow, fluctuation, imprecision, latency)
	end
end

# ╔═╡ 23041c74-867a-4990-9193-149bb006572d
# ╠═╡ skip_as_script = true
#=╠═╡
mechanics = OPMechanics()
  ╠═╡ =#

# ╔═╡ 1f052f6e-5676-4f66-aa9b-a3514bc20525
# Our action space
@enum PumpStatus off on

# ╔═╡ 01f7306b-d0de-4979-b0ca-9e5297ee5a16
PumpStatus

# ╔═╡ 23667919-c5b2-41d8-b4fd-92dd1c95e8cb
# Consumption rate
function consumption_rate(t)
	(t < 0) && error("Negative values of t ($t) unsupported.")
	t < 2 && return 0
	t < 4  && return 1.2
	t < 8 && return 0
	t < 10 && return 1.2
	t < 12 && return 2.5 
	t < 14 && return 0
	t < 16 && return 1.7
	t < 18 && return 0.5
	return 0
end

# ╔═╡ 3a1d3171-563e-48fa-b775-7c6fb6de02e3
# Next time the consumption rate changes
function next_rate_change(t)
	return t - t%2 + 2
end

# ╔═╡ ca873c15-677d-45f8-a5ee-c936de1b7094
md"""
The state is represented by the tuple $(t, v, p, l)$ where

 - ⁣$t$ is the time in the consumption cycle
 - ⁣$v$ is the volume of oil in the tank
 - ⁣$p$ is the pump status (Corresponding to the automaton locations *on* and *off*.)
 - ⁣$l$ is the latency-timer controlling how often the pump can switch state
"""

# ╔═╡ a2a0991c-7485-4bf3-8353-b33d2f4e9688
function simulate_point(mechanics::OPMechanics, 
		state, 
		action::PumpStatus, 
		random_outcomes)
	
	t, v, p, l = state # Time, Velocity, Pump-state, Latency-timer
	p ∉ [Int(on) Int(off)] && error("p $p out of range")
	t′, v′, p′, l′ = t, v, p, l

	if Int(action) != p && l′ <= 0.0
		p′ = Int(action)
		l′ = mechanics.latency
	end
	
	# fluctuation changes twice as often as the decision-period, 
	# so we compute the decision period in a fist and second half
	while t′ < t + mechanics.time_step/2
		time_step = min(t + mechanics.time_step/2 - t′, next_rate_change(t′) - t′)
		consumption = consumption_rate(t′)
		if consumption_rate(t′) > 0
			consumption += random_outcomes[1] # Fluctuation 1
		end
		v′ = v′ - consumption*time_step 
		t′ += time_step
	end
	while t′ < t + mechanics.time_step
		time_step = min(t + mechanics.time_step - t′, next_rate_change(t′) - t′)
		consumption = consumption_rate(t′)
		if consumption_rate(t′) > 0
			consumption += random_outcomes[2] # Fluctuation 2
		end
		v′ = v′ - consumption*time_step 
		t′ += time_step
	end
	
	t′ = t′%20
	
	l′ = round(l′ - mechanics.time_step, digits=10) # Floats, amirite?
	if p′ == Int(on)
		v′ += mechanics.inflow*mechanics.time_step
	end

	return t′, v′, p′, l′ # TODO: Imprecision
end

# ╔═╡ 5ed3810f-dedd-4844-a491-3a0ad3547b15
function simulate_point(mechanics::OPMechanics, state, action::PumpStatus)
	
	random_outcomes = [
		rand(-mechanics.fluctuation:0.001:mechanics.fluctuation),
		rand(-mechanics.fluctuation:0.001:mechanics.fluctuation)
	]
	
	simulate_point(mechanics::OPMechanics, state, action::PumpStatus, random_outcomes)
end

# ╔═╡ dead1cb7-09e8-45c5-81dd-8c3bb164741d
#=╠═╡
@bind t NumberField(0:mechanics.time_step:20., default=6)
  ╠═╡ =#

# ╔═╡ 9edd10ec-ec79-4bfa-8c04-2eaed8738335
#=╠═╡
consumption_rate(t)
  ╠═╡ =#

# ╔═╡ 96d053e5-0963-43c1-8b9a-82bcb021d5ef
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	plot(t -> consumption_rate(t), 
		xlim=(0, 20), 
		xticks=0:2:20,
		label="consumption",
		xlabel="time (s)",
		ylabel="rate (l/s)",
		line=(colors.PETER_RIVER, 3),
		yticks=[0, 0.5, 1.2, 1.7, 2.5])

	vline!([t], 
		line=colors.WET_ASPHALT,
		label="t (rate = $(consumption_rate(t)))")
end
  ╠═╡ =#

# ╔═╡ e7effef6-61e2-4b0d-bb94-75760ba8c00a
#=╠═╡
next_rate_change(t)
  ╠═╡ =#

# ╔═╡ d5c6d42a-3471-4005-80ec-9be2163962c9
#=╠═╡
@bind v0 NumberField(mechanics.v_min:0.1:mechanics.v_max)
  ╠═╡ =#

# ╔═╡ ae4052b1-31e9-424a-a144-6df0d17338f8
# ╠═╡ skip_as_script = true
#=╠═╡
@bind p Select([Int(p) for p in instances(PumpStatus)])
  ╠═╡ =#

# ╔═╡ 7f469b21-2d6c-4e6f-8aa9-939aa743a6a7
#=╠═╡
@bind l NumberField(0:0.1:mechanics.latency)
  ╠═╡ =#

# ╔═╡ a490bc63-a9d9-4261-aca8-0a1f877d09ce
# ╠═╡ skip_as_script = true
#=╠═╡
@bind action Select(instances(PumpStatus) |> collect)
  ╠═╡ =#

# ╔═╡ a214953e-b4d2-482c-aaad-7fc22a6b8feb
#=╠═╡
simulate_point(mechanics, (t, v0, p, l), action, [-0.1, -0.1])
  ╠═╡ =#

# ╔═╡ b2decea5-1a22-4711-8534-cc88cbabfb5c
#=╠═╡
consumption_rate(t)*mechanics.time_step
  ╠═╡ =#

# ╔═╡ db311b13-5ae0-4f24-b461-a64d62ea41cd
#=╠═╡
mechanics.inflow*mechanics.time_step
  ╠═╡ =#

# ╔═╡ 6c41264e-db32-4a5f-aa82-438a82824681
md"""
## A Full Trace

Notice I use a random agent.
"""

# ╔═╡ 126e00be-7899-405a-bd6e-731c76215726
struct OPTrace
	ts
	vs
	ps 
	ls
	elapsed
	actions
end

# ╔═╡ 77907f6b-4986-4699-838b-b849363416d2
initial_state = (0., 10., 0., 0.)

# ╔═╡ 5e2912de-768c-4711-82a9-d809f78e93ff
function simulate_trace(mechanics::OPMechanics, state, policy; duration=20)
	t, v, p, l = state
	ts, vs, ps, ls, elapsed, actions = [t], [v], [p], [l], [0.], []
	
	while elapsed[end] < duration
		a = policy((t, v, p, l))
		t, v, p, l = simulate_point(mechanics, (t, v, p, l), a)
		push!(ts, t)
		push!(vs, v)
		push!(ps, p)
		push!(ls, l)
		push!(elapsed, elapsed[end] + mechanics.time_step)
		push!(actions, a)
	end
	OPTrace(ts, vs, ps, ls, elapsed, actions)
end

# ╔═╡ 675538d0-5291-4069-9363-57464ba1012f
#=╠═╡
trace = simulate_trace(mechanics, 
		(t, v0, Int(off), 0.), 
		(_...) -> rand([on off]),
		duration = 20
	)
  ╠═╡ =#

# ╔═╡ a5c44feb-e5f5-4ce0-a911-7ad0c7bd4acf
#=╠═╡
# Nice syntax for unpacking structs
(;ts, vs, ps, ls, elapsed, actions) = trace
  ╠═╡ =#

# ╔═╡ 9e9174d8-7634-4a2f-ad07-f89e6af6fa3f
function plot_trace!(tup; params...)
	plot_trace!(tup...; params...)
end

# ╔═╡ 1a071438-946e-48ea-be42-189c1dc66bae
function plot_trace!(trace::OPTrace; show_actions=true, params...)
	(;vs, ts, ps, ls, elapsed, actions) = trace
	plot!(elapsed, vs,
		line=(colors.WET_ASPHALT, 3),
		label="tank",
		xlabel="time (s)",
		ylabel="volume (l)";
		params...)

	if show_actions
		scatter!([(t, v) for (t, v, p, l) in zip(elapsed, vs, ps, ls)
					if p == Int(on)],
			marker=(colors.PETER_RIVER, :utriangle, 6),
			markerstrokewidth=0,
			label="on")
		
		scatter!([(t, v) for (t, v, p, l) in zip(elapsed, vs, ps, ls)
					if p == Int(off)],
			marker=(colors.ORANGE, :dtriangle, 6),
			markerstrokewidth=0,
			label="off")
	end
	plot!()
end

# ╔═╡ 3b049ee6-f6c0-4b03-ab7b-aa5d7b72d387
#=╠═╡
begin
	plot()
	plot_trace!(trace, legend=:topleft)
	
	plot!(x -> consumption_rate((t + x)%mechanics.period), 
		label="consumption",
		line=(colors.PETER_RIVER, 3))
end
  ╠═╡ =#

# ╔═╡ a8d0fb4d-2a37-4c00-b14e-9ad929e75433
md"""
# Safety
"""

# ╔═╡ a775dad5-974c-4aae-b1dc-81535bf960cc
is_safe(state, m::OPMechanics) = m.v_min <= state[2] <= m.v_max

# ╔═╡ e362439c-716f-4dcb-91b4-3eaddceab0ea
function count_unsafe_traces(mechanics::OPMechanics, policy::Function; 
	runs=1000,
	run_duration=120)

	unsafe_count = 0
	unsafe_trace = nothing
	for i in 1:runs
		trace = simulate_trace(mechanics, initial_state, policy, duration=run_duration)
		(;ts, vs, ps, ls, elapsed, actions) = trace

		if !all([mechanics.v_min < v < mechanics.v_max for v in vs])
			unsafe_count += 1
			unsafe_trace = trace
		end
	end

	return (unsafe=unsafe_count, total=runs, unsafe_trace)
end

# ╔═╡ 64084492-f5f7-4f5c-b21c-8191d512f9c4
md"""
# Cost Function

Minimize the average accumulated oil volume in the limit, i.e. minimize

$\displaystyle\lim_{{{T}\to\infty}}\frac{1}{{T}}{\int_{{{t}={0}}}^{{T}}}{v}{\left({t}\right)}\text{d}{t}$

And since this notebook works on discretized time, I am rewriting it to be 

$\displaystyle\frac{\Delta_{{t}}}{{T}}{\sum_{{{i}={0}}}^{{{T}/\Delta_{{t}}}}}{v}{\left({i}\Delta_{{t}}\right)}$

Where $\Delta_t$ is the `time_step` variable from the mechanics.

"""

# ╔═╡ f1f39983-dd7e-4169-a3e3-1b63bc21ea2c
function cost(trace::OPTrace, time_step)
	(;ts, vs, ps, ls, elapsed, actions) = trace

	accumulator = 0
	for v in vs
		accumulator += v
	end

	(time_step/elapsed[end])*accumulator
end

# ╔═╡ b0ef9fe2-3cf8-413b-9950-328da3572d47
function cost(mechanics::OPMechanics, policy::Function; runs=1000, run_duration=120)
	initial_state = (0., 10., 0, 0.)
	accumulator = 0
	for i in 1:runs
		trace = simulate_trace(mechanics, initial_state, policy, 
			duration=run_duration)

		accumulator += cost(trace, mechanics.time_step)
	end

	accumulator/runs
end

# ╔═╡ 98c3de3f-9d27-4a5d-acb5-02c5b6f42f97
#=╠═╡
cost(mechanics, (_...) -> rand([on off]), run_duration=20, runs=100)
  ╠═╡ =#

# ╔═╡ 9830b7bd-4561-4dcb-9ca5-2b32da5ef132
#=╠═╡
cost(simulate_trace(mechanics, 
		(0., 10., 0, 0.), 
		(_...) -> rand([on off]),
		duration = 20
	), 
	mechanics.time_step)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═ca24fa88-c330-11ed-16b7-655126a8c62d
# ╠═25fa7e5a-04f2-49df-a0b0-0a3f69d721c9
# ╟─484bbefc-c519-4d8f-9f25-df8596deecd9
# ╟─7144e8f2-38af-4f32-8377-47841261bac5
# ╠═ebd20f9c-a927-4a50-98e4-2aec1f2c8e1b
# ╠═23041c74-867a-4990-9193-149bb006572d
# ╠═1f052f6e-5676-4f66-aa9b-a3514bc20525
# ╠═01f7306b-d0de-4979-b0ca-9e5297ee5a16
# ╠═23667919-c5b2-41d8-b4fd-92dd1c95e8cb
# ╠═9edd10ec-ec79-4bfa-8c04-2eaed8738335
# ╟─96d053e5-0963-43c1-8b9a-82bcb021d5ef
# ╠═3a1d3171-563e-48fa-b775-7c6fb6de02e3
# ╠═e7effef6-61e2-4b0d-bb94-75760ba8c00a
# ╟─ca873c15-677d-45f8-a5ee-c936de1b7094
# ╠═a2a0991c-7485-4bf3-8353-b33d2f4e9688
# ╠═5ed3810f-dedd-4844-a491-3a0ad3547b15
# ╠═dead1cb7-09e8-45c5-81dd-8c3bb164741d
# ╠═d5c6d42a-3471-4005-80ec-9be2163962c9
# ╠═ae4052b1-31e9-424a-a144-6df0d17338f8
# ╠═7f469b21-2d6c-4e6f-8aa9-939aa743a6a7
# ╠═a490bc63-a9d9-4261-aca8-0a1f877d09ce
# ╠═a214953e-b4d2-482c-aaad-7fc22a6b8feb
# ╠═b2decea5-1a22-4711-8534-cc88cbabfb5c
# ╠═db311b13-5ae0-4f24-b461-a64d62ea41cd
# ╟─6c41264e-db32-4a5f-aa82-438a82824681
# ╠═126e00be-7899-405a-bd6e-731c76215726
# ╠═77907f6b-4986-4699-838b-b849363416d2
# ╠═5e2912de-768c-4711-82a9-d809f78e93ff
# ╠═675538d0-5291-4069-9363-57464ba1012f
# ╠═a5c44feb-e5f5-4ce0-a911-7ad0c7bd4acf
# ╟─3b049ee6-f6c0-4b03-ab7b-aa5d7b72d387
# ╠═9e9174d8-7634-4a2f-ad07-f89e6af6fa3f
# ╠═1a071438-946e-48ea-be42-189c1dc66bae
# ╟─a8d0fb4d-2a37-4c00-b14e-9ad929e75433
# ╠═a775dad5-974c-4aae-b1dc-81535bf960cc
# ╠═e362439c-716f-4dcb-91b4-3eaddceab0ea
# ╟─64084492-f5f7-4f5c-b21c-8191d512f9c4
# ╠═f1f39983-dd7e-4169-a3e3-1b63bc21ea2c
# ╠═b0ef9fe2-3cf8-413b-9950-328da3572d47
# ╠═98c3de3f-9d27-4a5d-acb5-02c5b6f42f97
# ╠═9830b7bd-4561-4dcb-9ca5-2b32da5ef132
