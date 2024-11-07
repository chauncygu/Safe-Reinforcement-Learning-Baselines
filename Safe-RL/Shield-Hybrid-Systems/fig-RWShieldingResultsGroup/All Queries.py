import argparse
from os.path import exists

parser = argparse.ArgumentParser()

parser.add_argument("--results-dir", help="Warning: Contents of this folder will be moved to an Experiments Backup folder, overwriting that folder's previous content.")
parser.add_argument("--queries-models-dir", help="Directory containing the appropriate UPPAAL queries and associated models.")
parser.add_argument("--uppaal-dir", default="~/opt/uppaal-4.1.20-stratego-10-linux64/")
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

uppaaldir = args.uppaal_dir
resultsdir = args.results_dir
qmdir = args.queries_models_dir

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

    header = "Experiment;Runs;Deterrence;Avg. Cost;Avg. Deaths;Avg. Interventions"
    print(header)
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
        print(f"DROPPED INCONSISTENT ROW: {','.join(values)}")
        return
    
    row = [experiment, runs, deterrence, *values]
    results_csv = ";".join(row)
    print(results_csv)
    os.system(f"echo '{results_csv}' >> '{resultsdir}/Results.csv'")

re_mean = re.compile("mean=([\d.e-]+)")
re_aborted = re.compile("EXCEPTION: |is time-locked.|-- Aborted.")

def get_savedir(experiment, runs, iteration):
    return f"{resultsdir}/{iteration}/{experiment}/{runs}Runs" if runs != None else f"{resultsdir}/{iteration}/{experiment}"

def run_experiment(experiment, model, queries, runs, iteration):
    abspath_model = f"{qmdir}/{model}"
    abspath_queries = f"{qmdir}/{queries}"
    savedir = get_savedir(experiment, runs, iteration)
    runs = runs or 0
    os.system(f"mkdir -p '{savedir}'")
    
    # Command to run UPPAAL verifier
    command = f"{uppaaldir}/bin/verifyta -s --epsilon 0.001 --max-iterations 1 --good-runs {runs} --total-runs {runs} --runs-pr-state {runs} '{abspath_model}' '{abspath_queries}'"

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
        append_results(experiment, str(runs), extracted_queryresults)
    else:
        progress_update("QUERY ABORTED; DROPPED ROW. (Probably that bloody time-lock.)")



# Move the strategies into ther correct folder. 
# Can't be done in the run_experiment step since I need the unshielded strategies for the post-shielding experiment.
def cleanup_strategies(experiment, runs, iteration):
    savedir = get_savedir(experiment, runs, iteration)
    os.system(f"mv '{resultsdir}/'*.strategy.json '{savedir}'")



if __name__ == "__main__":
    if args.test:
        # HARDCODED: The number of iterations it re-runs the experiment.
        repeats = 1
        # HARDCODED: The number of training runs used to produce each strategy.
        learning_runs = [100, 200]
    else:
        # HARDCODED: The number of iterations it re-runs the experiment.
        repeats = 10
        # HARDCODED: The number of training runs used to produce each strategy.
        learning_runs = [1500, 3000, 6000, 12000]

    progress_update("Experiment started.")
    clear_results()

    for i in range(repeats):

        # No learning occurs in the Layabout model, so it is only run once.
        run_experiment( experiment = "Layabout",
                        model = "RW__ShieldedLayabout.xml",          # shield_enabled = true; layabout = true
                        queries = "ShieldedLayabout.q",    # Run the three queries without a strategy.
                        runs = None,
                        iteration = i)

        for runs in  learning_runs:

            run_experiment( experiment = "PreShielded",
                            model = "RW__PreShielded.xml",      # shield_enabled = true
                            queries = "PreShielded.q", # Train a strategy, save it, then evaluate it.
                            runs = runs,
                            iteration = i)

            cleanup_strategies("PreShielded", runs, i)
            
            run_experiment( experiment = "NoShield",
                            model = "RW__Unshielded.xml",    # shield_enabled = false
                            queries = "Unshielded.q", # Train a strategy, save it, then evaluate it.
                            runs = runs,
                            iteration = i)

            run_experiment( experiment = "PostShielded",
                            model = "RW__PostShielded.xml",      # shield_enabled = true
                            queries = "PostShielded.q",      # Load the previous strategy, then evaluate it.
                            runs = runs,
                            iteration = i)

            cleanup_strategies("NoShield", runs, i)


    progress_update("All done.")