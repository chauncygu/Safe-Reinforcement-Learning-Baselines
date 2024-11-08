//Load a strategy using  deterrence in {1000, 100, 10, 0}, then evaluate it.

/* formula 1 */
strategy Deterrence1000 = loadStrategy {} -> {x, t}  ("%resultsdir%/Deterrence1000.strategy.json")

/* formula 2 */
E[#<=30;%checks%] (max:total_cost) under Deterrence1000

/* formula 3 */
E[#<=30;%checks%00] (max:t>1) under Deterrence1000

/* formula 4 */
E[#<=30;%checks%] (max:100*interventions/(steps || 1)) under Deterrence1000

/* formula 5 */
strategy Deterrence100 = loadStrategy {} -> {x, t}  ("%resultsdir%/Deterrence100.strategy.json")

/* formula 6 */
E[#<=30;%checks%] (max:total_cost) under Deterrence100

/* formula 7 */
E[#<=30;%checks%00] (max:t>1) under Deterrence100

/* formula 8 */
E[#<=30;%checks%] (max:100*interventions/(steps || 1)) under Deterrence100

/* formula 9 */
strategy Deterrence10 = loadStrategy {} -> {x, t}  ("%resultsdir%/Deterrence10.strategy.json")

/* formula 10 */
E[#<=30;%checks%] (max:total_cost) under Deterrence10

/* formula 11 */
E[#<=30;%checks%00] (max:t>1) under Deterrence10

/* formula 12 */
E[#<=30;%checks%] (max:100*interventions/(steps || 1)) under Deterrence10


/* formula 13 */
strategy Deterrence0 = loadStrategy {} -> {x, t}  ("%resultsdir%/Deterrence10.strategy.json")

/* formula 14 */
E[#<=30;%checks%] (max:total_cost) under Deterrence0

/* formula 15 */
E[#<=30;%checks%00] (max:t>1) under Deterrence0

/* formula 16 */
E[#<=30;%checks%] (max:100*interventions/(steps || 1)) under Deterrence0

