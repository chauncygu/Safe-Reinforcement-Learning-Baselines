using Dates
using Glob
using Serialization
include("../Shared Code/ExperimentUtilities.jl")
include("../Shared Code/Cruise.jl")
include("../Shared Code/Squares.jl")
include("../Shared Code/CCBarbaricReachabilityFunction.jl")
include("../Shared Code/ShieldSynthesis.jl")


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
        return shield_dir
    end

    # If you want something done...
    progress_update("No shield was provided. Synthesising a new shield instead.")

    if test
        grid = get_cc_grid(1.0, ccmechanics)
        samples_per_axis = 1
    else
        grid = get_cc_grid(0.5, ccmechanics)
        samples_per_axis = 4
    end

    number_of_grid_points = length(grid_points(box(grid, [1, 1, 1]), samples_per_axis))
    
    progress_update("Synthesising shield: $number_of_grid_points samples $(grid.G) G")
    progress_update("Estimated time to synthesise: 35 minutes (3 minutes if run with --test)")

    shield_dir = working_dir ⨝ "$number_of_grid_points samples $(grid.G) G.shield"
    
    initialize!(grid, standard_initialization_function)

    reachability_function = get_barbaric_reachability_function(samples_per_axis, ccmechanics)
    shield, max_steps_reached = make_shield(reachability_function, CCAction, grid)
    robust_grid_serialization(shield_dir, shield)
    return shield_dir
end

"""
    compile_libccpreshield(working_dir, shield_dir, source_dir)

Load shield from `shield_dir`, and bake it into a `libccpreshield.so` file.
The C source code is copied from `source_dir` into `working_dir` 
where the given shield is written to `shield_dump.c` and compilation takes place.
Returns the path to `liccpreshield.so` which will be somewhere within `working_dir`.
"""
function compile_libccpreshield(working_dir, shield_dir, source_dir)
    shield = robust_grid_deserialization(shield_dir)

    # Bake it into the C-library. 
    # Make a copy first so we don't overwrite the original files
    for file in glob("*.c", source_dir)
        cp(file, working_dir ⨝ basename(file), force=true)
    end
    
    # Write to shield dump
    write(working_dir ⨝ "shield_dump.c", get_c_library_header(shield, "Retrieved with get_libccshield"))'
    
    previous_working_dir = pwd()
    cd(working_dir)
    
    # Good luck.

    # preshield
    run(`gcc -c -fPIC preshield.c -o preshield.o`)
    run(`gcc -shared -o libccpreshield.so preshield.o`)

    cd(previous_working_dir)

    return working_dir ⨝ "libccpreshield.so"
end

"""
    compile_libccpostshield(working_dir, shield_dir, source_dir, julia_config_dir)

Load shield from `shield_dir`, and bake it into a a `libccpostshield.so` file.
The C source code is copied from `source_dir` into `working_dir` 
where the given shield is written to `shield_dump.c` and compilation takes place.
Returns the path to `libccpostshield.so` which will be somewhere within `working_dir`.
"""
function compile_libccpostshield(working_dir, shield_dir, source_dir, julia_config_dir)
    shield = robust_grid_deserialization(shield_dir)

    # Bake it into the C-library. 
    # Make a copy first so we don't overwrite the original files
    for file in glob("*.c", source_dir)
        cp(file, working_dir ⨝ basename(file), force=true)
    end

    previous_working_dir = pwd()
    cd(working_dir)

    # postshield
    # Command to get compiler flags for gcc
    cmd_julia_config = Cmd([julia_config_dir, "--cflags", "--ldflags", "--ldlibs"])

    run(pipeline(
        cmd_julia_config, 
        `xargs gcc -c -fPIC  postshield.c -o postshield.o`)
    )

    run(pipeline(
        cmd_julia_config,
        `xargs gcc -shared -o libccpostshield.so postshield.o`)
    )

    cd(previous_working_dir)

    return working_dir ⨝ "libccpostshield.so"
end

function get_libccshield(possible_shield_file, source_dir; preshield_destination, postshield_destination, working_dir, julia_config_dir, test=false)
    # Getting shield
    shield_dir = get_shield(possible_shield_file, working_dir; test)

    preshield_location  = compile_libccpreshield(working_dir, shield_dir, source_dir)
    postshield_location = compile_libccpostshield(working_dir, shield_dir, source_dir, julia_config_dir)

    if preshield_destination != preshield_location
        cp(preshield_location, preshield_destination, force=true)
    end

    if postshield_destination != postshield_location
        cp(postshield_location, postshield_destination, force=true)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running as standalone script. This is suitable for testing.")
    working_dir = mktempdir()
    println("Working in: $working_dir")
    possible_shield_file = "../Export/CCShields/CC 375 Samples 0.5 G.shield"
    test = true
    preshield_destination = homedir() ⨝ "/libccshield.2.so"
    postshield_destination = homedir() ⨝ "/libccpostshield.2.so"
    source_dir = "CLibrary"
    result = get_libccshield(possible_shield_file, source_dir; preshield_destination, postshield_destination, working_dir, test)

    println("Result 👉 $result")
    print("Enter to quit > ")
    readline()
end