NEUTRAL_SQUARE = 1	# Either action allowable
FAST_SQUARE = 2		# Must take the :fast action
BAD_SQUARE = 3		# Risk of safety violation


struct Grid{T}
  G::Real
  x_min::Real
  x_max::Real
  y_min::Real
  y_max::Real
  x_count::Int
  y_count::Int
  array::Matrix{T}
end

function Grid(G, x_min, x_max, y_min, y_max)
  x_count::Int = ceil((x_max - x_min)/G)
  y_count::Int = ceil((y_max - y_min)/G)
  array = Matrix(undef, x_count, y_count)
  Grid(G, x_min, x_max, y_min, y_max, x_count, y_count, array)
end

# Makes the grid iterable, returning each square in turn.
Base.length(grid::Grid) = length(grid.array)
Base.size(grid::Grid) = size(grid.array)
Base.iterate(grid::Grid) = Square(grid, 1, 1), (2, 1)
Base.iterate(grid::Grid, state) = begin
	i, j = state
	if i > grid.x_count
		i = 1
		j += 1
		return Base.iterate(grid, (i, j))
	end
	if j > grid.y_count
		return nothing
	end
	Square(grid, i, j), (i+1, j)
end

Base.show(io::IO, ::MIME"text/plain", grid::Grid) = println(io, "Grid($(grid.G), $(grid.x_min), $(grid.x_max), $(grid.y_min), $(grid.y_max))")

struct Square
  grid::Grid
  ix::Int
  iy::Int
end


function box(grid::Grid, x, y)::Square
	if x < grid.x_min || x >= grid.x_max
		error("x value out of bounds.")
	end
	if y < grid.y_min || y >= grid.y_max
		error("x value out of bounds.")
	end

	ix = floor(Int, (x - grid.x_min)/grid.G) + 1
	iy = floor(Int, (y - grid.y_min)/grid.G) + 1
	Square(grid, ix, iy)
end


function bounds(square::Square)
	ix, iy = square.ix-1, square.iy-1
	x_min, x_max = square.grid.x_min, square.grid.x_max
	y_min, y_max = square.grid.y_min, square.grid.y_max
	G = square.grid.G
	Ixl, Iyl = G * ix + x_min, G * iy + y_min
	Ixu, Iyu = G * (ix+1) + x_min, G * (iy+1) + y_min
	Ixl, Ixu, Iyl, Iyu
end


function set_value!(square::Square, value)
	square.grid.array[square.ix, square.iy] = value
end


function get_value(square::Square)
	square.grid.array[square.ix, square.iy]
end


function clear(grid::Grid)
	for ix in 1:grid.x_count
		for iy in 1:grid.y_count
			grid.array[ix, iy] = 0
		end
	end
end


function initialize!(grid::Grid, value_function=(Ixl, Ixu, Itl, Itu) -> Itu > 1 ? BAD_SQUARE : NEUTRAL_SQUARE)
	for ix in 1:grid.x_count
		for iy in 1:grid.y_count
			square = Square(grid, ix, iy)
			set_value!(square, value_function(bounds(square)...))
		end
	end
end


function draw(grid::Grid; colors=[:white, :black], show_grid=false, color_labels=[])
	colors = cgrad(colors, length(colors), categorical=true)
	x_tics = grid.x_min:grid.G:grid.x_max
	y_tics = grid.y_min:grid.G:grid.y_max
	
	hm = heatmap(x_tics, y_tics, permutedims(grid.array, (2, 1)), c=colors, colorbar=nothing, aspect_ratio=:equal, clim=(1, length(colors)))

	if show_grid && length(grid.x_min:grid.G:grid.x_max) < 100
		vline!(grid.x_min:grid.G:grid.y_max, color="#afafaf", label=nothing)
		hline!(grid.y_min:grid.G:grid.y_max, color="#afafaf", label=nothing)
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

	return hm
end


function cover(grid, x_lower, x_upper, y_lower, y_upper)
	ix_lower = floor((x_lower - grid.x_min)/grid.G) + 1 # Julia indexes start at 1
	ix_upper = floor((x_upper - grid.x_min)/grid.G) + 1
	
	iy_lower = floor((y_lower - grid.y_min)/grid.G) + 1
	iy_upper = floor((y_upper - grid.y_min)/grid.G) + 1
	
	# Discard squares outside the grid dimensions
	ix_lower = max(ix_lower, 1)
	ix_upper = min(ix_upper, grid.x_count)
	
	iy_lower = max(iy_lower, 1)
	iy_upper = min(iy_upper, grid.y_count)
	
	[ (ix, iy)
		for ix in ix_lower:ix_upper
		for iy in iy_lower:iy_upper
	]
end

# My grids keep breaking because of the type circus I have.
# This is my own fault for not making a proper package. 
function robust_grid_serialization(file, grid::Grid)
	grid_as_tuple = (;grid.G,
					  grid.x_min, grid.x_max,
					  grid.y_min, grid.y_max,
					  grid.x_count, grid.y_count,
					  grid.array)
	serialize(file, grid_as_tuple)
end

function robust_grid_deserialization(file)
	f = deserialize(file)

	# Check that the imported file has the correct fields
	if length(fieldnames(typeof(f)) ∩ fieldnames(typeof(Grid(1, 0, 1, 0, 1)))) < 8
		throw(ArgumentError("The selected file does not have the correct format."))
	end
	
	grid = Grid(f.G, f.x_min, f.x_max, f.y_min, f.y_max)
	for square in grid
		set_value!(square, f.array[square.ix, square.iy])
	end
	
	grid
end	

function stringdump(grid)
	result = Vector{Char}(repeat('?', grid.x_count*grid.y_count))
	for iy in 1:grid.y_count
		for ix in 1:grid.x_count 
			square_value = get_value(Square(grid, ix, iy))
			if square_value == BAD_SQUARE
				result[ix + (iy-1)*grid.x_count] = 'r'
			elseif square_value == FAST_SQUARE
				result[ix + (iy-1)*grid.x_count] = 'b'
			else
				result[ix + (iy-1)*grid.x_count] = 'w'
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
const float x_min = $(grid.x_min);
const float x_max = $(grid.x_max);
const float y_min = $(grid.y_min);
const float y_max = $(grid.y_max);
const char grid[] = "$(stringdump(grid))";""";
end