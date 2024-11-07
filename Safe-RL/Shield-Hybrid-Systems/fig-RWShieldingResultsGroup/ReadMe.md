# How Reinforcement Learning for Random Walk is Affected by Shielding

This experiment applies to the Random Walk problem.

The experiment has the following configurations:

- **Shielded Layabout** The shield is applied to a basic "agent", which simply takes the most unsafe action no matter the input. For the Random Walk, this simply entails always going slow. This creates a strategy fully dictated by the shield. 
- **No Shield** An agent is trained with no shield applied, receiving a penalty d on runs that violate the safety property. 
- **Post-shielded** A shield is applied to the same learning agents, that were trained and evaluated in the No Shield model. This means that the agents have been trained without a shield, but are subsequently being shielded during the evaluation phase. 
- **Pre-shielded** The Q-learning agents were trained with the shield in place. If the shield intervenes, it apppears to the agent that it’s suggested action has the outcome of the shielded action.

Run from parent directory as 

	julia "fig-RWShieldingResultsGroup/Run Experiment"


!!! info "Tips:"

	Some additional cli args are supported. View with `--help`.
	
	View progress using `tree` by doing `sudo apt install tree && tree %resultsdir%`. 

## Results Folder Structure

The figures will be in the root folder. The raw data they were created from will be in `Query Results/Results.csv`. This folder also contains the raw query results from running every configuration. 

Sample `Query Results` folder structure: 

	├── 0
	│   ├── Layabout
	│   │   └── Layabout.queryresults.txt
	│   ├── NoShield
	│   │   └── 1500Runs
	│   │       ├── DeathCosts1000.strategy.json
	│   │       ├── DeathCosts100.strategy.json
	│   │       ├── DeathCosts10.strategy.json
	│   │       └── NoShield.queryresults.txt
	│   ├── PostShielded
	│   │   └── 1500Runs
	│   │       └── PostShielded.queryresults.txt
	│   └── PreShielded
	│       └── 1500Runs
	│           ├── PreShielded.queryresults.txt
	│           └── PreShielded.strategy.json
	├── 1
	│   ├── Layabout
	│   │   └── Layabout.queryresults.txt
	[...]

The folders named with numbers 0-9 each contain a repeat of the whole experiment.

Each repeat contains the names of different configurations in their root. 

The layabout configuration is only run once (since it does not involve training) so it only contains a text file with the query results.

The other configurations will contain a folder for each different number of training runs used. These folders contain query results along with the exported strategies, if applicable.

The NoShield configurations have strategies trained with different penalties for safety violations.

The PreShielded configurations do not see safety violations and therefore only train one strategy each.

The PostShielded configurations borrow their strategies from NoShield.

## Code folder structure

Everything is tied together in `Run Experiment.jl`. This file retrieves an appropriate shield using the script `Get libbbshield.jl` and then feeds this into the UPPAAL models and queries. 

Queries are exectued against the models using the Python script `All Queries.py` which is what produces the folder `Query Results` described above.

UPPAAL models and queries are found in the `Blueprints` folder. They need to have specific values replaced, marked by `%template variables%` before they can be run. This is because UPPAAL does not handle relative file paths in a consistent way. 

The UPPAAL models are identical, save for the following variations:

 - **RW__Unhielded.xml** : `shield_enabled = false; layabout = false;`
 - **RW__Shielded.xml** : `shield_enabled = true; layabout = false;`
 - **RW__ShieldedLayabout.xml** : `shield_enabled = true; layabout = true;`
 
 Query files are used along with a model to create the configurations.
 
 - **TrainSaveEvaluate.q** : using  deterrence in {1000, 100, 10, 0}, train a strategy, save it, then evaluate it.
 - **TrainSaveEvaluateSingle.q** : Train a single strategy, save it, then evaluate it.
 - **NoStrategyEvaluate.q** : Evaluate the queries with no strategy applied.
 - **LoadEvaluate.q** : Load a strategy using  deterrence in {1000, 100, 10, 0}, then evaluate it.

The combinations are the following:

 - **Layabout**:	"RW__ShieldedLayabout.xml",  "NoStrategyEvaluate.q"
 - **NoShield**:	"RW__Unshielded.xml",  "TrainSaveEvaluate.q"
 - **PreShielded**:	"RW__Shielded.xml",  "LoadEvaluate.q"
 - **PostShielded**:	"RW__Shielded.xml",  "TrainSaveEvaluateSingle.q"
