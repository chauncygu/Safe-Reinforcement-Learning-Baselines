struct Grid
    G::Real
    dimensions::Int
    lower_bounds::Vector{Real}
    upper_bounds::Vector{Real}
    size::Vector{Int}
    array
end

    
function Grid(G, lower_bounds, upper_bounds)
    dimensions = length(lower_bounds)
    
    if dimensions != length(upper_bounds)
        throw(ArgumentError("Inconsistent dimensionality"))
    end

    size = zeros(Int, dimensions)
    for (i, (lb, ub)) in enumerate(zip(lower_bounds, upper_bounds))
        size[i] = ceil((ub-lb)/G)
    end
    size
    
    array = zeros(Int8, (size...,))
    Grid(G, dimensions, lower_bounds, upper_bounds, size, array)
end

Base.show(io::IO, grid::Grid) = println(io, 
"Grid($(grid.G), $(grid.lower_bounds), $(grid.upper_bounds))")

# Makes the grid iterable, returning each square in turn.
Base.length(grid::Grid) = length(grid.array)

Base.size(grid::Grid) = size(grid.array)

struct Square
    grid::Grid
    indices::Vector{Int}
end

# Begin iteration.
# State is the indices of the previous square
Base.iterate(grid::Grid) = begin
    indices = ones(Int, grid.dimensions)
    square = Square(grid, indices)
    square, indices
end

Base.iterate(grid::Grid, state) = begin
    indices = copy(state)
    
    for dim in 1:grid.dimensions
        indices[dim] += 1
        if indices[dim] <= grid.size[dim]
            break
        else
            if dim < grid.dimensions
                indices[dim] = 1
                # Proceed to incrementing next row
            else
                return nothing
            end
        end
    end
    Square(grid, indices), indices
end

function box(grid::Grid, state)
	indices = zeros(Int, grid.dimensions)

	for dim in 1:grid.dimensions
		if !(grid.lower_bounds[dim] <= state[dim] < grid.upper_bounds[dim])
			throw(ArgumentError("State is out of bounds for this grid"))
		end
		
		indices[dim] = floor(Int, (state[dim] - grid.lower_bounds[dim])/grid.G) + 1
	end
	
	Square(grid, indices)
end

Base.in(state::Union{Vector, Tuple}, grid::Grid) = begin
	for dim in 1:grid.dimensions
		if !(grid.lower_bounds[dim] <= state[dim] < grid.upper_bounds[dim])
			return false
		end
	end
	
	return true
end

Base.in(square::Square, grid::Grid) = square.grid == grid

function bounds(square::Square)
	grid = square.grid
	G, lower_bounds = grid.G, grid.lower_bounds
	upper = [i*G + lower_bounds[dim] 
		for (dim, i) in enumerate(square.indices)]
	lower = [b - G for b in upper]
	lower, upper
end

# Iterable object which returns regularly-spaced points within a square.
struct grid_points
    square::Square
    per_axis::Number
end

Base.length(s::grid_points) = s.per_axis^s.square.grid.dimensions

Base.iterate(s::grid_points) = begin
    if s.per_axis - 1 < 0
        throw(ArgumentError("Samples per axis must be at least 1."))
    end

    lower, upper = bounds(s.square)
    spacing = s.square.grid.G/(s.per_axis - 1)
    # The iterator state  is (spacing, lower, upper, indices).
    # First sample always in the lower-left corner. 
    return lower, (spacing, lower, upper, zeros(Int, s.square.grid.dimensions))
end

Base.iterate(s::grid_points, state) = begin
    grid = s.square.grid
    spacing, lower, upper, indices = state
    indices = copy(indices)

    for dim in 1:grid.dimensions
        indices[dim] += 1
        if indices[dim] <= s.per_axis - 1
            break
        else
            if dim < grid.dimensions
                indices[dim] = 0
                # Proceed to incrementing next row
            else
                return nothing
            end
        end
    end

    sample = [i*spacing + lower[dim] for (dim, i) in enumerate(indices)]
    sample, (spacing, lower, upper, indices)
end

Base.in(s::Union{Vector, Tuple}, square::Square) = begin
	lower, upper = bounds(square)
	for dim in 1:length(s)
		if !(lower[dim] <= s[dim] < upper[dim])
			return false
		end
	end
	return true
end

function set_value!(square::Square, value)
	square.grid.array[square.indices...] = value
end

function set_values!(squares::Vector{Square}, value)
	for square in squares
		set_value!(square, value)
	end
end

function get_value(square::Square)
	square.grid.array[square.indices...]
end

function clear!(grid::Grid)
	for square in grid
		set_value!(square, 0)
	end
end

function initialize!(grid::Grid, value_function=(_) -> 1)
	for square in grid
		set_value!(square, value_function(bounds(square)...))
	end
end


function indexof(clause, list)
	result = []
	for (i, v) in enumerate(list)
		if clause(v)
			push!(result, i)
		end
	end
	result
end

function draw(grid::Grid, slice;
				colors=[:white, :black], 
				color_labels=[],
				show_grid=false, 
				plotargs...)
	
	colors = cgrad(colors, length(colors), categorical=true)

	if 2 != count((==(Colon())), slice)
		throw(ArgumentError("The slice argument should be an array of indices and exactly two colons. Example: [:, 10, :]"))
	end
	
	x, y = indexof((==(Colon())), slice)
	
	x_lower = grid.lower_bounds[x]
	x_upper = grid.upper_bounds[x]
	y_lower = grid.lower_bounds[y]
	y_upper = grid.upper_bounds[y]
	
	x_tics = x_lower:grid.G:x_upper
	y_tics = y_lower:grid.G:y_upper
	
	array = view(grid.array, slice...)
	array = transpose(array) # Transpose argument for heatmap() seems to be ignored.
	hm = heatmap(x_tics, y_tics, 
					array,
					c=colors,
					colorbar=nothing)

	if show_grid && length(grid.lower_bounds[x]:grid.G:grid.lower_bounds[x]) < 100
		
		vline!(grid.lower_bounds[x]:grid.G:grid.upper_bounds[x], 
				color=:gray, label=nothing)
		
		hline!(grid.lower_bounds[y]:grid.G:grid.upper_bounds[y], 
				color=:gray, label=nothing)
	end

	# Show labels
	if length(color_labels) > 0
		if length(color_labels) != length(colors)
			throw(ArgumentError("Length of argument color_labels does not match  number of colors."))
		end
		for (color, label) in zip(colors, color_labels)
			# Apparently shapes are added to the legend even if the list is empty
		    plot!(Float64[], Float64[], seriestype=:shape, 
		        label=label, color=color)
		end
	end

	plot!(;plotargs...) # Pass additional arguments to Plots.jl
end

function cover(grid::Grid, lower, upper)
	throw(Error("Not updated"))
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

# My grids keep breaking because of the type circus I have.
# This is my own fault for not making a proper package. 
function robust_grid_serialization(file, grid::Grid)	
	grid_as_tuple = (;grid.G,
					  grid.dimensions,
					  grid.lower_bounds, grid.upper_bounds,
					  grid.size,
					  grid.array)
	serialize(file, grid_as_tuple)
end

# ╔═╡ 861aacdf-e310-410d-a73f-1c6957f35073
function robust_grid_deserialization(file)
	f = deserialize(file)

	# Check that the imported file has the correct fields
	if length(fieldnames(typeof(f)) ∩ fieldnames(typeof(Grid(1, [0, 0], [3, 3])))) < 6
		throw(ArgumentError("The selected file does not have the correct format."))
	end
	
	grid = Grid(f.G, f.lower_bounds, f.upper_bounds)
	for square in grid
		set_value!(square, f.array[square.indices...])
	end
	
	grid
end

# Offset for the ascii table. The zero value is offset to A and so on all the way to Z at 90.
# After this comes [ \ ] ^ _ and ` 
# Why these are here I don't know, but then come the lower case letters.
char_offset = 65

function stringdump(grid)
	result = Vector{Char}(repeat('?', prod(grid.size)))

	for (i, v) in enumerate(grid.array)
		result[i] = Char(v + char_offset)
	end
	
	return String(result)
end

function get_c_library_header(grid::Grid, description)
	arrayify(x) = "{ $(join(x, ", ")) }"
	
	lower_bounds = arrayify(grid.lower_bounds)
	upper_bounds = arrayify(grid.upper_bounds)
	size = arrayify(grid.size)
	"""
/* This code was automatically generated by function get_c_library_header*/
const int char_offset = $char_offset;
const char grid[] = "$(stringdump(grid))";
const float G = $(grid.G);
const int dimensions = $(grid.dimensions);
const int size[] = $(size);
const float lower_bounds[] = $(lower_bounds);
const float upper_bounds[] = $(upper_bounds);
// Description: $description """
end