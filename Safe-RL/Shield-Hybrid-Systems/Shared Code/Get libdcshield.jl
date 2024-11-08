# Plots required by BBShieldSynthesis. fml.
using Plots
using Serialization
using GridShielding
include("../Shared Code/ExperimentUtilities.jl")
include("../Shared Code/DC-DC Converter.jl")
include("../Shared Code/DCShielding.jl")

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

    m = DCMechanics()
    if test
        grid = get_dc_grid(m, 0.1)
        samples_per_axis = [2, 2, 1] # This won't be a safe shield
    else
        grid = get_dc_grid(m, 0.01)
        samples_per_axis = [4, 4, 1]
    end

    simulation_model = SimulationModel(get_simulation_function(m), get_randomness_space(m), samples_per_axis)
    
    progress_update("Synthesising shield: $(prod(samples_per_axis)) samples $(grid.granularity) G")
    progress_update("Estimated time to synthesise: 70 minutes (2 minutes if run with --test)")

    shield_dir = working_dir ⨝ "$(prod(samples_per_axis)) samples $(grid.granularity) G.shield"
    
    reachability_function = get_barbaric_reachability_function(simulation_model)
   shield, terminated_early = make_shield(reachability_function, SwitchStatus, grid)
    terminated_early && @warn "Shield didn't finish synthesising. This is surprising. The default max iterations is very large, so maybe it is in the process of very slowly colouring the whole play-space red."
    robust_grid_serialization(shield_dir, shield)
    return shield_dir
end

"""
    compile_libdcshield(working_dir, shield_dir, lib_source_code_dir)

Load shield from `shield_dir`, and bake it into a `libopshield.so` file.
The C source code is copied from `lib_source_code_dir` into `working_dir` 
where the given shield is written to `shield_dump.c` and compilation takes place.
Returns the path to `libopshield.so` which will be somewhere within `working_dir`.
"""
function compile_libdcshield(working_dir, shield_dir, lib_source_code_dir)
    shield = robust_grid_deserialization(shield_dir)

    # Copy over to our working directory.
    cp(lib_source_code_dir ⨝ "shield.c", working_dir ⨝ "shield.c", force=true)
    
    # Write to shield dump
    write(working_dir ⨝ "shield_dump.c", get_c_library_header(shield, "Retrieved with get_libopshield"))'
    
    previous_working_dir = pwd()
    cd(working_dir)
    
    # Good luck.
    run(`gcc -c -fPIC shield.c -o shield.o`)
    run(`gcc -shared -o libopshield.so shield.o`)

    cd(previous_working_dir)
    working_dir ⨝ "libopshield.so"
end

function get_libdcshield(possible_shield_file, lib_source_code_dir, lib_destination_dir; working_dir, test=false)
    
    # Getting shield
    shield_dir = get_shield(possible_shield_file, working_dir; test)

    lib_destination_dirʹ = compile_libdcshield(working_dir, shield_dir, lib_source_code_dir)
    if lib_destination_dir != lib_destination_dirʹ
        cp(lib_destination_dirʹ, lib_destination_dir, force=true)
    end
end
