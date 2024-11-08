//Load a strategy using  deterrence in {1000, 100, 10, 0}, then evaluate it.

/* formula 1 */
strategy Deterrence1000 = loadStrategy {} -> {p, v}("%resultsdir%/Deterrence1000.strategy.json")

/* formula 2 */
E[<=120;%checks%] (max:LearnerPlayer.fired) under Deterrence1000

/* formula 3 */
E[<=120;%checks%] (max:(number_deaths > 0)) under Deterrence1000

/* formula 4 */
E[<=120;%checks%] (max:interventions) under Deterrence1000

/* formula 5 */
strategy Deterrence100 = loadStrategy {} -> {p, v}("%resultsdir%/Deterrence100.strategy.json")

/* formula 6 */
E[<=120;%checks%] (max:LearnerPlayer.fired) under Deterrence100

/* formula 7 */
E[<=120;%checks%] (max:(number_deaths > 0)) under Deterrence100

/* formula 8 */
E[<=120;%checks%] (max:interventions) under Deterrence100

/* formula 9 */
strategy Deterrence10 = loadStrategy {} -> {p, v}("%resultsdir%/Deterrence10.strategy.json")

/* formula 10 */
E[<=120;%checks%] (max:LearnerPlayer.fired) under Deterrence10

/* formula 11 */
E[<=120;%checks%] (max:(number_deaths > 0)) under Deterrence10

/* formula 12 */
E[<=120;%checks%] (max:interventions) under Deterrence10

/* formula 13 */
strategy Deterrence0 = loadStrategy {} -> {p, v}("%resultsdir%/Deterrence0.strategy.json")

/* formula 14 */
E[<=120;%checks%] (max:LearnerPlayer.fired) under Deterrence0

/* formula 15 */
E[<=120;%checks%] (max:(number_deaths > 0)) under Deterrence0

/* formula 16 */
E[<=120;%checks%] (max:interventions) under Deterrence0


