#= Requirements:
using Dates
using Glob
=#


"""
	@nbparam(name, default)

Returns the value `default`, unless the dictionary `NBPARAMS` is defined and has the key `name`.

This should allow pluto notebooks to receive parameters when they are run as a script. 

I could have been better at wrapping my notebooks' functionality in, you know, functions. But bere we are."""
macro nbparam(name, default)
	if !@isdefined(NBPARAMS) || !haskey(NBPARAMS, name)
		return default
	else
		return NBPARAMS[name]
	end
end


# ("name.of.file.csv", 1) -> "name.of.file (1).csv"
function insert_suffix(file_name, suffix)
	split_period = split(file_name, ".")
	# Insert suffix last
	if length(split_period) == 1
		return file_name * suffix
	else
		# Insert suffix before the last period.
		return string(
			join(split_period[1:end-1], "."), 
			suffix, ".", 
			split_period[end])
	end
end

function insert_prefix(file_name, prefix)
	return prefix*file_name
end

function get_unique_file(path, file_name)
    if !isfile(joinpath(path, file_name)) 
        return joinpath(path, file_name)
    end
    
	prefix = 0
	result = ""
	while true
		result = joinpath(path, 
			insert_prefix(file_name, "($prefix) "))
		prefix += 1

		# Exit condition
		if !isfile(result) 
			break
		end
	end
	result
end

function progress_update(message)
	timestamp = Dates.format(now(), "HH:MM")
	println("$timestamp $message")
end

# Parse command line arguments into a dict in the way I recon it's done.
# Example: --foo bar -b baz -jkl 2 = Dict(foo => bar, b => baz, j => nothing, k => nothing, l => 2)
# TODO: Delete this, and use ArgParse library instead. Code kept to allow smooth transition.
function my_parse_args(args)::Dict
	one_letter_flag = r"^-\w"
	multi_letter_flag = r"^--\w"
	
	result = Dict()

	flag = nothing
	for arg in args
		if occursin(multi_letter_flag, arg)
			flag = arg[3:end]
			result[flag] = nothing
		elseif occursin(one_letter_flag, arg)
			for c in arg[2:end]
				flag = string(c)
				result[flag] = nothing
			end
		else
			# TODO: Restructure so I can check for duplicate assignment
			result[flag] = arg
		end
	end
	result
end

function search_and_replace(input_dir, output_dir, replacements, glob_pattern="*")
	if !isdir(output_dir)
		throw(error("Provided output dir does not exist: $output_dir"))
	end
	for filename in glob(glob_pattern, input_dir)
		
		file = filename |> read |> String
		outfile = joinpath(output_dir, basename(filename))
		
		open(outfile, "w") do io
			for line in split(file, "\n")
				line′ = replace(line, replacements...)
				println(io, line′)
			end
		end
	end
end

# Use to check that everything's been serach-and-replace'd.
# r"%[a-zA-Z_ ]+%" to match %template_variables%
function error_on_regex_in(dir, expression, glob_pattern="*")
	for filename in glob(glob_pattern, dir)
		
		file = filename |> read |> String
		line_number = 1
		for line in split(file, "\n")
			m = match(expression, line)
			if m != nothing
				error("Pattern matched: $line\n$filename:$line_number")
			end
			line_number += 1
		end
	end
end


"""     latex_table(df; [headers, group_by]) 

Turn a data-frame into a latex-table.

Args:

	group_by: Visually groups table by inserting double-hline between groups. Hack: Does not alter sorting. Table should already be sorted so grouped rows are together. On change in value of any column in the list, apply dobule hline between rows """
function latex_table(df; headers=nothing, group_by=[])
	if headers === nothing
		headers = [string(k) for k in keys(first(df))]
	end

	# \begin
	result = "\\begin{tabular}"
	# Those { c c c c } thingies
	result *= "{$(repeat("@{\\hspace{3mm}} c ", length(headers)))} \n"
	result *= "\\toprule\n"
	
	# Table header
	h = ["\\textbf{$cell}" for cell in headers]
	result *= join(h, "\t&\t")
	result *= "\\\\\n"
	

	# If we are grouping by column values, the header should have its own visual group
	if group_by != [] 
		result *= "\\midrule\n"
	end

	# The rows. The main feature.
	for (i, row) in enumerate(eachrow(df))
		r = [string(cell) for cell in row]
		result *= join(r, "\t&\t")
		
		result *= "\t\\\\\n"  	# \t \\ \n

		# Check wheter to apply double hline (if the column changes next row)
		if group_by != [] && i != nrow(df)
			next_values = [df[i+1, :][column] for column in group_by]
			if any([v != row[g] for (v, g) in zip(next_values, group_by)])
				result *= "\\midrule\n"
				continue
			end
		end
	end

	# \end
	result *= "\\end{tabular}"
	return result
end
	



function export_figure(results_dir, name, figure)
    savefig(figure, results_dir ⨝ "$name.png")
    savefig(figure, results_dir ⨝ "$name.svg")
    progress_update("Saved $name")
end

function export_table(results_dir, name, table)
    CSV.write(results_dir ⨝ "$name.csv", table)
    write(results_dir ⨝ "$name.txt", "$table")
    progress_update("Saved $name")
end