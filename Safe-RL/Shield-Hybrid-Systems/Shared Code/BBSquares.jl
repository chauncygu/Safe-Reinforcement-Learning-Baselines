using Plots

struct Grid
    G::Real
    v_min::Real
    v_max::Real
    p_min::Real
    p_max::Real
    v_count::Int
    p_count::Int
    array
end

	
function Grid(G, v_min, v_max, p_min, p_max)
    v_count::Int = ceil((v_max - v_min)/G)
    p_count::Int = ceil((p_max - p_min)/G)
    array = zeros(Int8, (v_count, p_count))
    Grid(G, v_min, v_max, p_min, p_max, v_count, p_count, array)
end

# Makes the grid iterable, returning each square in turn.
Base.length(grid::Grid) = length(grid.array)
Base.size(grid::Grid) = size(grid.array)
Base.iterate(grid::Grid) = Square(grid, 1, 1), (2, 1)
Base.iterate(grid::Grid, state) = begin
	i, j = state
	if i > grid.v_count
		i = 1
		j += 1
		return Base.iterate(grid, (i, j))
	end
	if j > grid.p_count
		return nothing
	end
	Square(grid, i, j), (i+1, j)
end

struct Square
    grid::Grid
    iv::Int
    ip::Int
end

Base.show(io::IO, ::MIME"text/plain", grid::Grid) = println(io, "Grid($(grid.G), $(grid.v_min), $(grid.v_max), $(grid.p_min), $(grid.p_max))")

# Iterable object which returns regularly-spaced points within a square.
struct grid_points
	square::Square
	per_axis::Number
end
Base.length(s::grid_points) = s.per_axis^2
Base.iterate(s::grid_points) = begin
	if s.per_axis - 1 < 0
		throw(ArgumentError("Samples per axis must be at least 1."))
	end

	Ivl, Ivu, Ipl, Ipu = bounds(s.square)
	spacing = s.square.grid.G/(s.per_axis - 1)
	# First sample always in the lower-left corner. 
	# The iterator state  is (spacing, i, j).
	return (Ivl, Ipl), (spacing, 1, 0)
end
Base.iterate(s::grid_points, state) = begin
	Ivl, Ivu, Ipl, Ipu = bounds(s.square)
	spacing, i, j = state
	
	v = Ivl + spacing*i
	p = Ipl + spacing*j

	if i > s.per_axis - 1
		i = 0
		j += 1
		return Base.iterate(s, (spacing, i, j))
	end
	if j > s.per_axis - 1
		return nothing
	end
	
	return (v, p), (spacing, i+1, j)
end


function box(grid::Grid, v, p)::Square
	if v < grid.v_min || v >= grid.v_max
		error("v value out of bounds.")
	end
	if p < grid.p_min || p >= grid.p_max
		error("p value out of bounds.")
	end

	iv = floor(Int, (v - grid.v_min)/grid.G) + 1
	ip = floor(Int, (p - grid.p_min)/grid.G) + 1
	Square(grid, iv, ip)
end


function bounds(square::Square)
	iv, ip = square.iv-1, square.ip-1
	v_min, v_max = square.grid.v_min, square.grid.v_max
	p_min, p_max = square.grid.p_min, square.grid.p_max
	G = square.grid.G
	Ivl, Ipl = G * iv + v_min, G * ip + p_min
	Ivu, Ipu = G * (iv+1) + v_min, G * (ip+1) + p_min
	Ivl, Ivu, Ipl, Ipu
end


function set_value!(square::Square, value)
	square.grid.array[square.iv, square.ip] = value
end

function set_values!(squares::Vector{Square}, value)
	for square in squares
		square.grid.array[square.iv, square.ip] = value
	end
end

function get_value(square::Square)
	square.grid.array[square.iv, square.ip]
end


function clear!(grid::Grid)
	for iv in 1:grid.v_count
		for ip in 1:grid.p_count
			grid.array[iv, ip] = 0
		end
	end
end


function initialize!(grid::Grid, value_function=
								(Ivl, Ivu, Ipl, Ipu) -> Ivl == 0 && Ipl == 0 ? 2 : 1)
	for iv in 1:grid.v_count
		for ip in 1:grid.p_count
			square = Square(grid, iv, ip)
			set_value!(square, value_function(bounds(square)...))
		end
	end
end


function draw(grid::Grid; 
				colors=[:white, :black], 
				color_labels=[],
				show_grid=false, 
				plotargs...)
	
	
	colors = cgrad(colors, length(colors), categorical=true)
	x_tics = grid.v_min:grid.G:grid.v_max
	y_tics = grid.p_min:grid.G:grid.p_max
	
	hm = heatmap(x_tics, y_tics, 
					transpose(grid.array), 
					c=colors,
					colorbar=nothing)

	if show_grid && length(grid.v_min:grid.G:grid.v_max) < 100
		vline!(grid.v_min:grid.G:grid.v_max, color=:gray, label=nothing)
		hline!(grid.p_min:grid.G:grid.p_max, color=:gray, label=nothing)
	end

	# Show labels
	if length(color_labels) > 0
		if length(color_labels) != length(colors)
			throw(ArgumentError("Length of argument color_labels does not match  number of colors."))
		end
		for (color, label) in zip(colors, color_labels)
		    plot!(Float64[], Float64[], seriestype=:shape,   # The Float typing has been added to avoid a weird warning
		        label=label, color=color)
		end
	end

	plot!(;plotargs...) # Pass additional arguments to Plots.jl
end

# For performance reasons, cover does not return proper squares, only their indexes.
function cover(grid, v_lower, v_upper, p_lower, p_upper)
	iv_lower = floor((v_lower - grid.v_min)/grid.G) + 1 # Julia indexes start at 1
	iv_upper = floor((v_upper - grid.v_min)/grid.G) + 1
	
	ip_lower = floor((p_lower - grid.p_min)/grid.G) + 1
	ip_upper = floor((p_upper - grid.p_min)/grid.G) + 1
	
	# Discard squares outside the grid dimensions
	iv_lower = max(iv_lower, 1)
	iv_upper = min(iv_upper, grid.v_count)
	
	ip_lower = max(ip_lower, 1)
	ip_upper = min(ip_upper, grid.p_count)
	
	[ (iv, ip)
		for iv in iv_lower:iv_upper
		for ip in ip_lower:ip_upper
	]
end

"""
`draw_diff(grid1, grid2, colors, labels)`

Draws a plot showing the differences between grid1 and grid2. 

Args: 

 - `diffcolors` List of colors to use when there is a difference
 - `bbshieldcolors` List of colors to use when values are identical.
 - `labels` List of labels. It should be the case that for all squares from both grids, `labels[get_value(square) - 1]` is defined.
"""
function draw_diff(grid1, grid2, diffcolors, bbshieldcolors, labels; name1=nothing, name2=nothing, plotargs...)
	diffcolors, bbshieldcolors = [diffcolors...], [bbshieldcolors...]
	diff_grid = Grid(grid1.G, grid1.v_min, grid1.v_max, grid1.p_min, grid1.p_max)
	i = length(bbshieldcolors)
	# Dict of label => (value, color)
	value_and_color::Dict{String, Tuple{Number, Colorant}} = Dict()

	# Returns the value of the corresponding label. Adds to the dict value_and_color
	get_or_add(label, colors) = if haskey(value_and_color, label)
		value_and_color[label][1] # return the value
	else
		if length(colors) == 0 
			error("Ran out of diffcolors. Please provide more.")
		end
		value_and_color[label] = (i, popfirst!(colors))
		i += 1
		value_and_color[label][1]
	end

	combine_labels(l1, l2) = "$l1 : $l2"

	# HACK: I am tired
	for (i, color) in enumerate(bbshieldcolors)
		label = labels[i]
		label = combine_labels(label, label)
		value_and_color[label] = (i-1, color)
	end
	
	for diff_square in diff_grid
		square1 = corresponding(diff_square, grid1)
		square2 = corresponding(diff_square, grid2)

		if isnothing(square1) || isnothing(square2)
			value = get_or_add("missing", diffcolors)
			set_value!(diff_square, value)
			continue
		end
		
		value1, value2 = get_value(square1), get_value(square2)
		label1, label2 = labels[value1 + 1], labels[value2 + 1]

		if value1 == value2
			label = combine_labels(label1, label2)
			value, _ = value_and_color[label]
			set_value!(diff_square, value)
		else
			label = combine_labels(label1, label2)
			value = get_or_add(label, diffcolors)
			set_value!(diff_square, value)
		end
	end

	

	colors_used = [ (i, c) for (i, c) in values(value_and_color) ]
	sort!(colors_used, by=x -> x[1])
	colors_used = map(x -> x[2], colors_used)

	labels = [ l for l in value_and_color]
	sort!(labels, by=x -> x[2][1])
	labels = map(x -> x[1], labels)
	
	draw(diff_grid, colors=colors_used, color_labels=labels; plotargs...)

	if name1 != nothing
		plot!([0], [0], linealpha=0, markeralpha=0, label=combine_labels(name1, name2))
	end
end;

function corresponding(square::Square, other::Grid)
	ivl, ivu, ipl, ipu = bounds(square)
	G = square.grid.G
	v, p = ivl + G/100, ipl + G/100
	if (v, p) ∈ other
		box(other, v, p)
	else
		nothing
	end
end

same_bounds(s::Square, s′::Square) = all([
	abs(b - b′) < 1e-10
	for (b, b′) in zip(bounds(s), bounds(s′))
])



# My grids keep breaking because of the type circus I have.
# This is my own fault for not making a proper package. 
function robust_grid_serialization(file, grid::Grid)
	grid_as_tuple = (;grid.G,
					  grid.v_min, grid.v_max,
					  grid.p_min, grid.p_max,
					  grid.v_count, grid.p_count,
					  grid.array)
	serialize(file, grid_as_tuple)
end

function robust_grid_deserialization(file)
	f = deserialize(file)

	# Check that the imported file has the correct fields
	if length(fieldnames(typeof(f)) ∩ fieldnames(typeof(Grid(1, 0, 1, 0, 1)))) < 8
		throw(ArgumentError("The selected file does not have the correct format."))
	end
	
	grid = Grid(f.G, f.v_min, f.v_max, f.p_min, f.p_max)
	for square in grid
		set_value!(square, f.array[square.iv, square.ip])
	end
	
	grid
end	


Base.in(state::Union{Vector, Tuple}, grid::Grid) = begin
	if !(grid.v_min <= state[1] < grid.v_max)
		return false
	end
	if !(grid.p_min <= state[2] < grid.p_max)
		return false
	end
	
	return true
end


Base.in(square::Square, grid::Grid) = square.grid == grid


Base.in(s::Union{Vector, Tuple}, square::Square) = begin
	ivl, ivu, ipl, ipu = bounds(square)
	
	if !(ivl <= s[1] < ivu)
		return false
	end
	
	if !(ipl <= s[2] < ipu)
		return false
	end
	
	return true
end

function stringdump(grid)
	result = Vector{Char}(repeat('?', grid.v_count*grid.p_count))
	for ip in 1:grid.p_count
		for iv in 1:grid.v_count
			square_value = get_value(Square(grid, iv, ip))
			if square_value == 2
				result[iv + (ip-1)*grid.v_count] = 'r'
			elseif square_value == 1
				result[iv + (ip-1)*grid.v_count] = 'b'
			else
				result[iv + (ip-1)*grid.v_count] = 'w'
			end
		end
	end
	return String(result)
end

function get_c_library_header(grid, description)
	"""
/* This code was automatically generated by function get_c_library_header*/
// $description
const float G = $(grid.G);
const float x_min = $(grid.v_min);
const float x_max = $(grid.v_max);
const float y_min = $(grid.p_min);
const float y_max = $(grid.p_max);
const char grid[] = "$(stringdump(grid))";""";
end