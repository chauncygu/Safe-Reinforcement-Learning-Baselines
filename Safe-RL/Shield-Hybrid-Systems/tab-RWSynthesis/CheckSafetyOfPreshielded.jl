# import Pkg
# Pkg.activate(".")

using Glob
using Serialization
include("../Shared Code/ExperimentUtilities.jl")
include("../Shared Code/Get librwshield.jl")

# infix operator "\join" redefined to signify joinpath
⨝ = joinpath

function compile_all_librwshield(working_dir, lib_source_code_dir)
    created_files = String[]
    skipped_files = String[]
    for shield_file in glob("*.shield", working_dir)
        if !shield_is_valid(robust_grid_deserialization(shield_file))
            push!(skipped_files, shield_file)
            continue
        end
        librwshield_file = compile_librwshield(working_dir, shield_file, lib_source_code_dir)
        updated_location = working_dir ⨝ "$(basename(shield_file)).so"
        mv(librwshield_file, updated_location, force=true)
        push!(created_files, updated_location)
    end
    # Cleanup all .c and .o files
    [rm(f) for f in glob("*.[oc]", working_dir)]

    return created_files, skipped_files
end

# "1 Samples 0.01 G.shield.so" => "1 Samples 0.01 G"
function just_shield_description(filename)
    result = basename(filename)
    return replace(result, ".shield.so" => "")
end

function check_shields(compiled_shields, working_dir, blueprints_dir, uppaal_dir; test=false, just_print_the_commands=false)
    shield_dirs = String[] # Working dir for the evaluation of each shield
    for librwshield_file in compiled_shields
        progress_update("Checking shield $(basename(librwshield_file))")
        shield_dir = working_dir ⨝ just_shield_description(librwshield_file)
        rm(shield_dir, recursive=true, force=true)
        mkdir(shield_dir)

        if !isdir(blueprints_dir)
            throw(error("Blueprints folder not found. Make sure this script is exectued from the root of the code folder.\nCurrent directory: $(pwd())\nContents: $(readdir())"))
        end

        replacements = Dict(
            "%resultsdir%" => shield_dir,
            "%rwshieldfile%" => librwshield_file
        )

        search_and_replace(blueprints_dir, shield_dir, replacements)
        push!(shield_dirs, shield_dir)
    end

    training_runs = test ? "100" : "12000"
    alpha =   test ? "0.1" : "0.01"
    beta =    test ? "0.1" : "0.01"
    epsilon = test ? "0.05" : "0.000001"
    
    for shield_dir in shield_dirs
        verifyta = uppaal_dir ⨝ "bin/verifyta.sh"
        model = shield_dir ⨝ "RW__PreShielded.xml"
        queries = shield_dir ⨝ "TrainSaveCheckSafety.q"

        @assert isfile(model)
        @assert isfile(queries)
        
        cmd = Cmd(String[verifyta, model, queries,# Usage: verifyta MODEL QUERY [OPTION]...
            "--silence-progress",                 #   -s [ --silence-progress ]            Do not display the progress indicator.
            "--max-iterations", "1",             #   --max-iterations arg                 Maximal total number of iterations in the learning algorithm
            "--good-runs", training_runs,        #   --good-runs arg                      Use <number> good runs for each learning
            "--total-runs", training_runs,       #   --total-runs arg                     Number of total runs to attempt for  learning
            "--runs-pr-state", training_runs,    #   --runs-pr-state arg                  Number of good runs stored per discrete  state
            "--epsilon", epsilon,                #   -E [ --epsilon ] arg                 probability uncertainty (epsilon).
            "--alpha", alpha,                    #   -a [ --alpha ] arg                   probability of false negatives (alpha).
            "--beta", beta,                      #   -B [ --beta ] arg                    probability of false positives (beta).
            ])

        if just_print_the_commands # This may be useful for running the queries on the cluster
            println(cmd)
        else
            write(shield_dir ⨝ "query_results.txt", read(cmd))
        end
    end

    return shield_dirs
end

# "[...] \n (0/17 runs) Pr(<> ...) in [0,0.195064] (95% CI)" ==> "[0,0.195064] (95% CI)"
function extract_result(query_results)
    m = match(r"\((?<unsafe>\d+)/(?<total>\d+) runs\) Pr\(<> ...\) in (?<smc_result>.+\(\d+% CI\))", query_results)
    m === nothing && error("Did not find the relevant query result.")
    return (m[:unsafe], m[:total], m[:smc_result])
end

function read_query_results(shield_dirs, result_file)
    open(result_file, "w") do io
        println(io, "file;label;safety_violations_observed;runs;fraction_unsafe;smc_result")
        for shield_dir in shield_dirs
            shield_description = basename(shield_dir)
            query_results = read(shield_dir ⨝ "query_results.txt", String)

            unsafe, total, smc_result = extract_result(query_results)
            fraction_unsafe = parse(Int, unsafe)/parse(Int, total)
            row = join([
                    "\"$shield_description\"",
                    "preshielded",
                    unsafe,
                    total,
                    fraction_unsafe,
                    "\"$smc_result\""],
                ";")

            println(io, row)
        end
    end;
end

function append_skipped_shields(skipped_shields, result_file)
    open(result_file, "a") do io
        for skipped in skipped_shields
            shield_description = replace(basename(skipped), ".shield" => "")
            unsafe = total = fraction_unsafe = 1
            smc_result = "\"-\""
            row = join([
                    shield_description,
                    "skipped",
                    unsafe,
                    total,
                    fraction_unsafe,
                    smc_result],
                ";")

            println(io, row)
        end
    end
end

"""
    check_safety_of_preshielded(;shields_dir, results_dir, lib_source_code_dir, blueprints_dir, uppaal_dir, test, just_print_the_commands=false)

**Args**
 - `shields_dir` Directory containing the exported strategies. Probably called `Exported Strategies`
 - `results_dir` Will be used as working directory and contain the results.
 - `lib_source_code_dir` Directory which contains the `shield.c` source file.
 - `blueprints_dir` Directory which contains blueprints for the UPPAAL model.
 - `uppaal_dir` Directory which contains `bin/verifyta`
 - `test` Produce useless results, but quickly.
 - `just_print_the_commands` Don't actually run the queries, just set everything up and print the commands to do so in the console. Useful for putting the queries onto the cluster.
"""
function check_safety_of_preshielded(;shields_dir, results_dir, lib_source_code_dir, blueprints_dir, uppaal_dir, test, just_print_the_commands=false)
       
    compiled_shields, skipped_shields = compile_all_librwshield(shields_dir, lib_source_code_dir)
    query_result_dirs = check_shields(compiled_shields, results_dir, blueprints_dir, uppaal_dir; test, just_print_the_commands)
    read_query_results(query_result_dirs, results_dir ⨝ "Evaluations.csv")
    append_skipped_shields(skipped_shields, results_dir ⨝ "Evaluations.csv")
end

# Tests written during development. You'll have to comment in the usings to run them.
if abspath(PROGRAM_FILE) == @__FILE__
    #compiled_shields = compile_all_librwshield(homedir() ⨝ "Results/tab-RWSynthesis/Exported Strategies/", homedir() ⨝ "ReproducibilityPackage/Shared Code/librwshield")

    #compiled_shields = [homedir() ⨝ "Results/tab-RWSynthesis/Exported Strategies/1 Samples 0.01 G.shield.so",
    #                    homedir() ⨝ "Results/tab-RWSynthesis/Exported Strategies/1 Samples 0.02 G.shield.so",
    #                    homedir() ⨝ "Results/tab-RWSynthesis/Exported Strategies/1 Samples 0.05 G.shield.so",
    #                    homedir() ⨝ "Results/tab-RWSynthesis/Exported Strategies/4 Samples 0.01 G.shield.so"]

    #shield_dirs = check_shields(compiled_shields, homedir() ⨝ "Results/tab-RWSynthesis/Evaluations", "./tab-RWSynthesis/Blueprints", homedir() ⨝ "opt/uppaal-4.1.20-stratego-10-linux64/", test=true, just_print_the_commands=false)

    # shield_dirs = [homedir() ⨝ "Results/tab-RWSynthesis/Evaluations/1 Samples 0.01 G",
    #                homedir() ⨝ "Results/tab-RWSynthesis/Evaluations/1 Samples 0.02 G",
    #                homedir() ⨝ "Results/tab-RWSynthesis/Evaluations/1 Samples 0.05 G",
    #                homedir() ⨝ "Results/tab-RWSynthesis/Evaluations/4 Samples 0.01 G"]

    #read_query_results(shield_dirs, homedir() ⨝ "Results/tab-RWSynthesis/Evaluations/Evaluations.csv")

    shields_dir = homedir() ⨝ "Results/tab-RWSynthesis/Exported Strategies"
    results_dir = homedir() ⨝ "Results/tab-RWSynthesis/Evaluations"
    lib_source_code_dir = homedir() ⨝ "A1/ReproducibilityPackage/Shared Code/librwshield"
    blueprints_dir = "./tab-RWSynthesis/Blueprints"
    uppaal_dir = homedir() ⨝ "opt/uppaal-4.1.20-stratego-10-linux64/"
    test = true
    just_print_the_commands = false
    check_safety_of_preshielded(;shields_dir, results_dir, lib_source_code_dir, blueprints_dir, uppaal_dir, test, just_print_the_commands)
end