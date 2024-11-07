import argparse
from os.path import exists

parser = argparse.ArgumentParser()

# OBS: These default-values are NOT used in the experiment. The defaults are actually set in Run Experiment.jl
parser.add_argument("--results-dir", help="Warning: Contents of this folder will be moved to an Experiments Backup folder, overwriting that folder's previous content.")
parser.add_argument("--queries-models-dir", help="Directory containing the appropriate UPPAAL queries and associated models.")
parser.add_argument("--uppaal-dir", default="~/opt/uppaal-4.1.20-stratego-11-linux64/")
parser.add_argument("--strategy-for-postshield", help="Full path to file where the strategy should be saved which will be post-shielded.")
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

uppaaldir = args.uppaal_dir
resultsdir = args.results_dir
qmdir = args.queries_models_dir
postshieldme = args.strategy_for_postshield

# HARDCODED: Number of runs to train the intervention strategy of a post-shield. 
post_shield_tweak_runs_multiplier = 1/3

import os
import re
from datetime import datetime

def progress_update(progress):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp}\t{progress}")

def clear_results():
    backupdir = f"{resultsdir}/../Query Results Backup"

    if exists(backupdir):
        os.system(f"rm -rd '{backupdir}'")
    os.system(f"mkdir '{backupdir}'") 
    os.system(f"mv '{resultsdir}/'* '{backupdir}'")

    header = "Experiment;Runs;Deterrence;Avg. Cost;Avg. Crashes;Avg. Interventions"
    progress_update(f"Add Row: {header}")
    os.system(f"echo '{header}' > '{resultsdir}/Results.csv'")


# As you can see in clear_results, a row consists of the experiment done, the number of runs, the cost of death and then the results: average swings, deaths and interventions.
# A query file will either have a non-applicable cost of death, or it will spit out results for all three variations at once.
# So if I get 9 values, that means its the results for the 3 tiers of what a death costs. 
def append_results(experiment, runs, values, deterrence="-"):
    if len(values) == 4*3:
        append_results(experiment, runs, values[0:3], deterrence="1000")
        append_results(experiment, runs, values[3:6], deterrence="100")
        append_results(experiment, runs, values[6:9], deterrence="10")
        append_results(experiment, runs, values[9:12], deterrence="0")
        return
    elif len(values) != 3:
        progress_update(f"DROPPED INCONSISTENT ROW: {','.join(values)}")
        return
    
    row = [experiment, runs, deterrence, *values]
    results_csv = ";".join(row)
    progress_update(f"Add Row: {results_csv}")
    os.system(f"echo '{results_csv}' >> '{resultsdir}/Results.csv'")

re_mean = re.compile("mean=([\d.e-]+)")
re_aborted = re.compile("EXCEPTION: |is time-locked.|-- Aborted.")

def get_savedir(experiment, runs, iteration):
    return f"{resultsdir}/{iteration}/{experiment}/{runs}Runs" if runs != None else f"{resultsdir}/{iteration}/{experiment}"

def run_experiment(experiment, model, queries, runs, iteration, deterrence="-", post_shield_tweak_runs=None):
    savedir = get_savedir(experiment, runs, iteration)
    runs = runs or 0
    os.system(f"mkdir -p '{savedir}'")

    # When learning an intervention-strategy for a post-shield, use a constant amount of runs.
    # However, the number that goes in the csv is how many runs the post-shielded strategy was originally trained for.
    actual_runs = post_shield_tweak_runs or runs

    abspath_model = f"{qmdir}/{model}"
    abspath_queries = f"{qmdir}/{queries}"
    
    # Command to run UPPAAL verifier
    command = f"{uppaaldir}/bin/verifyta -s --epsilon 0.001 --max-iterations 1 --good-runs {actual_runs} --total-runs {actual_runs} --runs-pr-state {actual_runs} '{abspath_model}' '{abspath_queries}'"

    progress_update(f"Running: {command}")

    queryresults = f"{savedir}/{experiment}.queryresults.txt"

    # Save the query so we know what we just ran
    os.system(f"cat '{abspath_queries}' > '{queryresults}'")
    # Run the command and save append it to the queryresults file.
    os.system(f"{command} >> '{queryresults}' 2>&1") 

    # Do regex on the queryresults and save the mean values using append_results.
    # Some kind of time lock occurs which messes up the results. If a time lock happens, I have to discard that query
    abort = False
    extracted_queryresults = []
    with open(queryresults, "r") as f:
        for line in f:
            extracted_queryresults += re_mean.findall(line)
            if re.search(re_aborted, line):
                abort = True
                break
    if not abort:
        append_results(experiment, str(runs), extracted_queryresults, deterrence)
    else:
        progress_update("QUERY ABORTED; DROPPED ROW. (Probably that bloody time-lock.)")



# Move the strategies into ther correct folder. 
# Can't be done in the run_experiment step since I need the unshielded strategies for the post-shielding experiment.
def cleanup_strategies(experiment, runs, iteration):
    savedir = get_savedir(experiment, runs, iteration)
    os.system(f"mv '{resultsdir}/'*.strategy.json '{savedir}'")

def run_post_shield_experiments(deterrence):
    run_experiment( experiment = f"PostShieldedRandomChoice",
                    model = "CC__PostShieldedNondeterministic.xml",  # Nondeterministic: If the post-shielded strategy picks an unsafe action, the remaining safe actions can be selected in the UPPAAL model.
                    queries = "NoStrategyEvaluate.q",                # Evaluate the model under no particular strategy. (randomly choose a different action when the shield intervenes)
                    runs = runs,
                    iteration = i,
                    deterrence = deterrence,
                    post_shield_tweak_runs = int(runs*post_shield_tweak_runs_multiplier))

    run_experiment( experiment = f"PostShieldedPolicyPreferred",
                    model = "CC__PostShieldedDeterministic.xml",  # Deterministic: If the post-shielded strategy picks an unsafe aciton, the next best action is taken, according to the same strategy.
                    queries = "NoStrategyEvaluate.q",             # Evaluate the model under no particular strategy. (The strategy is provided by postshield.c)
                    runs = runs,
                    iteration = i,
                    deterrence = deterrence,
                    post_shield_tweak_runs = int(runs*post_shield_tweak_runs_multiplier))

    run_experiment( experiment = f"PostShieldedInterventionMinimized",
                    model = "CC__PostShieldedNondeterministic.xml",  # Nondeterministic: If the post-shielded strategy picks an unsafe action, the remaining safe actions can be selected in the UPPAAL model.
                    queries = "MinimizeInterventionsEvaluate.q",     # Minimize interventions, then evaluate the model.
                    runs = runs,
                    iteration = i,
                    deterrence = deterrence,
                    post_shield_tweak_runs = int(runs*post_shield_tweak_runs_multiplier))

    run_experiment( experiment = f"PostShieldedCostMinimized",
                    model = "CC__PostShieldedNondeterministic.xml",  # Nondeterministic: If the post-shielded strategy picks an unsafe action, the remaining safe actions can be selected in the UPPAAL model.
                    queries = "MinimizeCostEvaluate.q",              # Minimize cost, then evaluate the model.
                    runs = runs,
                    iteration = i,
                    deterrence = deterrence,
                    post_shield_tweak_runs = int(runs*post_shield_tweak_runs_multiplier))




if __name__ == "__main__":
    if args.test:
        # HARDCODED: The number of iterations it re-runs the experiment.
        repeats = 1
        # HARDCODED: The number of training runs used to produce each strategy.
        learning_runs = [10, 20]
    else:
        # HARDCODED: The number of iterations it re-runs the experiment.
        repeats = 10
        # HARDCODED: The number of training runs used to produce each strategy.
        learning_runs = [1500, 3000, 6000, 12000]

    progress_update("Experiment started.")
    clear_results()

    for i in range(repeats):

        for runs in  learning_runs:

            run_experiment( experiment = "PreShielded",
                            model = "CC__Shielded.xml",            # Shield active during learning
                            queries = "TrainSaveEvaluateSingle.q", # Train a strategy, save it, then evaluate it.
                            runs = runs,
                            iteration = i)

            cleanup_strategies("PreShielded", runs, i)
            
            run_experiment( experiment = "NoShield",
                            model = "CC__Unshielded.xml",    # Original cruise (with egoVelocityMin = -8 though)
                            queries = "TrainSaveEvaluate.q", # Train strategies with different penalties, save them, then evaluate them.
                            runs = runs,
                            iteration = i)

            
            # Take unshielded strategy and write it to the strategy-file that will be post-shielded.
            os.system(f"cp '{resultsdir}/Deterrence1000.strategy.json' '{postshieldme}'")

            run_post_shield_experiments(deterrence=f"{1000}")

            # Take unshielded strategy and write it to the strategy-file that will be post-shielded.
            os.system(f"cp '{resultsdir}/Deterrence100.strategy.json' '{postshieldme}'")

            run_post_shield_experiments(deterrence=f"{100}")

            # Take unshielded strategy and write it to the strategy-file that will be post-shielded.
            os.system(f"cp '{resultsdir}/Deterrence10.strategy.json' '{postshieldme}'")

            run_post_shield_experiments(deterrence=f"{10}")

            # Take unshielded strategy and write it to the strategy-file that will be post-shielded.
            os.system(f"cp '{resultsdir}/Deterrence0.strategy.json' '{postshieldme}'")

            run_post_shield_experiments(deterrence=f"{0}")


            cleanup_strategies("NoShield", runs, i)


    progress_update("All done.")