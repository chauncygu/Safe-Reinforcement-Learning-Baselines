# Plots required by BBShieldSynthesis. fml.
using Plots
using Serialization
using Glob
using ProgressLogging
include("../Shared Code/ExperimentUtilities.jl")
include("../Shared Code/RandomWalk.jl")
include("../Shared Code/RWSquares.jl")
include("../Shared Code/RWShieldSynthesis.jl")


# infix operator "\join" redefined to signify joinpath
⨝ = joinpath

"""
    get_shield(possible_shield_file, working_dir, [test])

 Retrieve a shield, put a copy in working_dir and return its absolute path.
 If `possible_shield_file` is `nothing`, a new shield will be synthesised. 
 Otherwise, the file at that path will be used."""
function get_shield(possible_shield_file, working_dir; test)
    if possible_shield_file !== nothing
        if !isfile(possible_shield_file)
            error("File not found: $possible_shield_file")
        end
        progress_update("Using existing shield found at $possible_shield_file")
        shield_dir = working_dir ⨝ basename(possible_shield_file)
        if possible_shield_file != shield_dir
            cp(possible_shield_file, shield_dir, force=true)
        end
        progress_update("Saved a copy to $shield_dir")
        return shield_dir
    end

    # If you want something done...
    progress_update("No shield was provided. Synthesising a new shield instead.")

    if test
        grid = Grid(0.005, 0, 1.2, 0, 1.2)
    else
        grid = Grid(0.005, 0, 1.2, 0, 1.2)
    end
    
    progress_update("Synthesising shield: RW $(grid.G) G")
    progress_update("Estimated time to synthesise: 1 minute")

    shield_dir = working_dir ⨝ "RW $(grid.G) G.shield"
    
    initialize!(grid)

    shield, terminated_early, animation = make_shield(rwmechanics..., grid)
    robust_grid_serialization(shield_dir, shield)
    progress_update("Saved shield to $shield_dir")
    return shield_dir
end

"""
    compile_librwshield(working_dir, shield_dir, lib_source_code_dir)

Load shield from `shield_dir`, and bake it into a `librwshield.so` file.
The C source code is copied from `lib_source_code_dir` into `working_dir` 
where the given shield is written to `shield_dump.c` and compilation takes place.
Returns the path to `libbshield.so` which will be somewhere within `working_dir`.
"""
function compile_librwshield(working_dir, shield_dir, lib_source_code_dir)
    shield = robust_grid_deserialization(shield_dir)

    # Bake it into the C-library. 
    # Make a copy first so we don't overwrite the original files
    for file in glob(lib_source_code_dir ⨝ "*")
        cp(file, working_dir ⨝ basename(file), force=true)
    end

    # Write to shield dump
    write(working_dir ⨝ "shield_dump.c", get_c_library_header(shield, "Retrieved with get_librwshield"))'
    
    previous_working_dir = pwd()
    cd(working_dir)
    
    # Good luck.
    run(`gcc -c -fPIC shield.c -o shield.o`)
    run(`gcc -shared -o librwshield.so shield.o`)

    cd(previous_working_dir)
    working_dir ⨝ "librwshield.so"
end

function get_librwshield(possible_shield_file, lib_source_code_dir, lib_destination_dir; working_dir, test=false)
    
    # Getting shield
    shield_dir = get_shield(possible_shield_file, working_dir; test)

    lib_destination_dirʹ = compile_librwshield(working_dir, shield_dir, lib_source_code_dir)
    if lib_destination_dir != lib_destination_dirʹ
        cp(lib_destination_dirʹ, lib_destination_dir, force=true)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    possible_shield_file = homedir() ⨝ "/Results/tab-RWSynthesis/Exported Strategies/25 Samples 0.02 G.shield"
    test = true
    lib_destination_dir = homedir() ⨝ "/librwshield.2.so"
    lib_source_code_dir = "N/A" # Removed because it was nonsense
    println("Running as standalone script. This is suitable for testing.")
    result = get_librwshield(possible_shield_file, lib_source_code_dir, lib_destination_dir; test)

    println("Running result 👉 $result")
    print("Enter to quit > ")
    readline()
end