# Experiment E - Effects of Shielding on a Learning Agent

 
## Results Folder Structure

The figures will be in the root folder. The raw data they were created from will be in `Query Results/Results.csv`. This folder also contains the raw query results from running every configuration. 

Sample `Query Results` folder structure: 

	├── 0
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

All configurations will contain a folder for each different number of training runs used. These folders contain query results along with the exported strategies, if applicable.

The NoShield configurations have strategies trained with different penalties for safety violations.

The PreShielded configurations do not see safety violations and therefore only train one strategy each.

The PostShielded configurations borrow their strategies from NoShield.

## Note on Shielding 

Like in other variants of this experiment, pre-shielding is done by compiling a shared-object file from C code, which the shield has been "baked into." That is, it has been exported as a very large string literal which is included in the binary. 

However, post-shielding is handled differently, to enable different ways of choosing between remaining safe actions, when the shield intervenes. Post Shielding logic is resolved using the notebook PostShield Strategy.jl. It loads a UPPAAL strategy from given file path, parses it, and applies it along with a shield that is loaded similarly. 

This notebook is called in UPPAAL using a bridge written in C. The C code makes call to julia as described in [this article.](https://docs.julialang.org/en/v1/manual/embedding/)

It is by far the most brittle code of all my experiments, and that says a lot. UPPAAL is required to load both this bridging C code, *and* the associated julia-to-c library, `<julia.h>`. I simply got lucky that on my machine, DLL hell was avoided by calling the `verifyta` binary directly. 

When a post-shielded strategy picks an unsafe action, there will in some cases be two safe actions remaining. For example, picking *forwards* might be potentially unsafe, but both *neutral* and *backwards* are okay. In this instance, which of the two should be picked? The experiment tries several options.

- **PostShieldedRandomChoice** : Choose randomly between remaining options.
- **PostShieldedPolicyPreferred** : Choose whichever remaining option gives the best expected outcome, according to the strategy. 
- **PostShieldedInterventionMinimized** : Train a second stategy to choose in these cases. Optimize it to minimize the number of future interventions.
- **PostShieldedCostMinimized** : Train a second stategy to choose in these cases. Optimize it to minimize the cost, just like the original strategy.

## Code Folder Structure

Everything is tied together in `Run Experiment.jl`. This file retrieves an appropriate shield using the script `Get libccshield.jl` and then feeds this into the UPPAAL models and queries. 

`Get libccshield` uses the source C files in `libcc` to compile the shared object files that will be used by the UPPAAL models.

UPPAAL models and queries are found in the `Blueprints` folder. They need to have specific values replaced, marked by `%template variables%` before they can be run. This is because UPPAAL does not handle relative file paths in a consistent way. 

Queries are exectued against the models using the Python script `All Queries.py` which is what produces the folder `Query Results` described above. 

Each UPPAAL model is different:

 - **CC__Unhielded.xml** : Is close to the original version.
 - **CC__Shielded.xml** : Uses `libccpreshield.so` to restrict unsafe edges.
 - **CC__PostShieldedDeterministic.xml** : Uses `libccpostshielded.so` to restrict edges, so that there is only ever one allowed edge. This edge is the one dictated by *PostShieldedPolicyPreferred*
 - **CC__PostShieldedNondeterministic.xml** : Uses `libccpostshielded.so` to restrict edges, so that there is only a choice between safe actions, in the case where the strategy itself did not pick a safe action. When the strategy picks a safe action, only that edge is allowed.
 
 Query files are used along with a model to create the configurations.
 
 - **TrainSaveEvaluate.q** : using  deterrence in {1000, 100, 10, 0}, train a strategy, save it, then evaluate it.
 - **TrainSaveEvaluateSingle.q** : Train a single strategy, save it, then evaluate it.
 - **NoStrategyEvaluate.q** : Evaluate the queries with no strategy applied.
 - **LoadEvaluate.q** : Load a strategy using  deterrence in {1000, 100, 10, 0}, then evaluate it.

The combinations are the following:

 - **NoShield**:	"CC__Unhielded.xml",  "TrainSaveEvaluate.q"
 - **PreShielded**:	"CC__Shielded.xml",  "LoadEvaluate.q"
 - **PostShieldedRandomChoice**:	"CC__PostShieldedDeterministic.xml",  "TrainSaveEvaluateSingle.q"
 - **PostShieldedPolicyPreferred**:	"CC__PostShieldedNondeterministic.xml",  "TrainSaveEvaluateSingle.q"
 - **PostShieldedInterventionMinimized**:	"CC__PostShieldedDeterministic.xml",  "MinimizeInterventionsEvaluate.q"
 - **PostShieldedCostMinimized**:	"CC__PostShieldedDeterministic.xml",  "MinimizeCostEvaluate.q"
