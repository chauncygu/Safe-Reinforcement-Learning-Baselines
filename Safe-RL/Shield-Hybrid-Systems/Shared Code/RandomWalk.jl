rwmechanics = (ϵ = 0.04, δ_fast = 0.17, δ_slow = 0.1, τ_fast = 0.05, τ_slow = 0.12)

function step(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, x, t, a; unlucky=false)
	x′, t′ =  x, t
	if unlucky
		if a == :fast
			x′ = x + δ_fast - ϵ
			t′ = t + τ_fast + ϵ
		else
			x′ = x + δ_slow - ϵ
			t′ = t + τ_slow + ϵ
		end
	else
		if a == :fast
			x′ = x + rand(δ_fast - ϵ:0.005:δ_fast + ϵ )
			t′ = t + rand(τ_fast - ϵ:0.005:τ_fast + ϵ )
		else
			x′ = x + rand(δ_slow - ϵ:0.005:δ_slow + ϵ )
			t′ = t + rand(τ_slow - ϵ:0.005:τ_slow + ϵ )
		end
	end	
	
	x′, t′
end


function plot_with_size(x_max, t_max; figure_width=600, figure_height=600)
	plot(	xlim=[0, x_max],
			ylim=[0, t_max], 
			aspectratio=:equal, 
			xlabel="x",
			ylabel="t",
			size=(figure_width, figure_height))
	hline!([x_max], c=:gray)
	vline!([t_max], c=:gray)
end


function plot_with_size!(x_max, t_max; figure_width=600, figure_height=600)
	plot!(	xlim=[0, x_max],
			ylim=[0, t_max], 
			aspectratio=:equal, 
			xlabel="x",
			ylabel="t",
			size=(figure_width, figure_height))
	hline!([x_max], c=:gray)
	vline!([t_max], c=:gray)
end


function draw_next_step!(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, 
			x, t, a;
			colors=(slow=:yellow, fast=:blue, line=:black))
	if a == :both
		draw_next_step!(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, x, t, :fast, colors=colors)
		return draw_next_step!(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, x, t, :slow, colors=colors)
	end
	color = a == :fast ? colors.fast : colors.slow
	linestyle = a == :fast ? :solid : :dash
	scatter!([x], [t], 
		markersize=3, 
		markerstrokewidth=0,
		color=colors.line)
	δ, τ = a == :fast ? (δ_fast, τ_fast) : (δ_slow, τ_slow)
	δ, τ = δ + x, τ + t
	plot!(Shape([δ - ϵ, δ - ϵ, δ + ϵ, δ + ϵ], 
				[τ - ϵ, τ + ϵ, τ + ϵ, τ - ϵ]), 
			color=color,
			opacity=0.8,
			linewidth=0,
			legend=nothing)
	plot!([x, δ], [t, τ], linestyle=linestyle, linewidth=1, linecolor=color)
end


function draw_walk!(xs, ts, actions;
	colors=(slow=:yellow, fast=:blue, line=:black))
linecolors = [a == :fast ? colors.fast : colors.slow for a in actions]
linestyles = [a == :fast ? :solid : :dash for a in actions]
push!(linecolors, colors.line)
push!(linestyles, :solid)
plot!(xs, ts,
markershape=:circle,
markersize=3,
markercolor=colors.line,
markerstrokewidth=0,
linewidth=3,
linecolor=linecolors,
linestyle=linestyles,
legend=nothing)
end


function take_walk(	cost_function, cost_of_losing,
					x_max, t_max, 
					ϵ, δ_fast, δ_slow, τ_fast, τ_slow, 
					policy::Function;
					unlucky=false)
	xs, ts, actions = [0.], [0.], []
	total_cost = 0

	while last(xs) < x_max && last(ts) < t_max
		a = policy(last(xs), last(ts))
		x, t = step(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, 
					last(xs), last(ts), a, unlucky=unlucky)
		total_cost += cost_function(last(xs), last(ts), a)
		push!(xs, x)
		push!(ts, t)
		push!(actions, a)
	end

	winner = last(ts) < t_max

	if !winner
		total_cost += cost_of_losing
	end

	(;xs, ts, actions, total_cost, winner)
end


function evaluate(	cost_function, cost_of_losing, 
					x_max, t_max, 
					ϵ, δ_fast, δ_slow, τ_fast, τ_slow, 
					policy::Function; iterations=1000)
	losses = 0
	costs = []
	for i in 1:iterations
		_, _, _, cost, winner = take_walk(	cost_function, cost_of_losing,
											x_max, t_max, 
											ϵ, δ_fast, δ_slow, τ_fast, τ_slow, 
											policy)
		if !winner
			losses += 1
		end
		push!(costs, cost)
	end
	perfect = losses == 0
	average_wins = (iterations - losses) / iterations
	average_cost = sum(costs)/length(costs)
	(;perfect, average_wins, average_cost)
end

function draw(policy::Function, x_max, t_max, G; colors=[:blue, :yellow])
	size_x, size_t = Int(x_max/G), Int(t_max/G)
	matrix = Matrix(undef, size_x, size_t)
	for i in 1:size_x
		for j in 1:size_t
			x, t = i*G, j*G
			matrix[i, j] = policy(x, t) == :fast ? 1 : 2
		end
	end

	x_tics = 0:G:x_max
	y_tics = 0:G:t_max
	
	heatmap(x_tics, y_tics, matrix, 
			c=colors, 
			colorbar=nothing, 
			aspect_ratio=:equal, 
			clim=(1, length(colors)))
end