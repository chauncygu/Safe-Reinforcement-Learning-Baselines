//using  deterrence in {1000, 100, 10, 0}, train a strategy, save it, then evaluate it.
// HACK: Since this query file is only used for NoShield, I don't bother estimating the number of interventions. It will be zero, but I want to keep that number.

/* formula 1 */
strategy Deterrence1000 = minE (total_cost + (t>1)*1000) [#<=30] {} -> {x, t} : <> x>=1 or t>=1

/* formula 2 */
saveStrategy("%resultsdir%/Deterrence1000.strategy.json", Deterrence1000)

/* formula 3 */
E[#<=30;%checks%] (max:total_cost) under Deterrence1000

/* formula 4 */
E[#<=30;%checks%00] (max:t>1) under Deterrence1000

/* formula 5 */
E[#<=30;%checks%] (max:100*interventions/(steps || 1)) under Deterrence1000

/* formula 6 */
strategy Deterrence100 = minE (total_cost + (t>1)*100) [#<=30] {} -> {x, t} : <> x>=1 or t>=1

/* formula 7 */
saveStrategy("%resultsdir%/Deterrence100.strategy.json", Deterrence100)

/* formula 8 */
E[#<=30;%checks%] (max:total_cost) under Deterrence100

/* formula 9 */
E[#<=30;%checks%00] (max:t>1) under Deterrence100

/* formula 10 */
E[#<=30;%checks%] (max:100*interventions/(steps || 1)) under Deterrence100

/* formula 11 */
strategy Deterrence10 = minE (total_cost + (t>1)*10) [#<=30] {} -> {x, t} : <> x>=1 or t>=1

/* formula 12 */
saveStrategy("%resultsdir%/Deterrence10.strategy.json", Deterrence10)

/* formula 13 */
E[#<=30;%checks%] (max:total_cost) under Deterrence10

/* formula 14 */
E[#<=30;%checks%00] (max:t>1) under Deterrence10

/* formula 15 */
E[#<=30;%checks%] (max:100*interventions/(steps || 1)) under Deterrence10

/* formula 16 */
strategy Deterrence0 = minE (total_cost + (t>1)*0) [#<=30] {} -> {x, t} : <> x>=1 or t>=1

/* formula 17 */
saveStrategy("%resultsdir%/Deterrence0.strategy.json", Deterrence0)

/* formula 18 */
E[#<=30;%checks%] (max:total_cost) under Deterrence0

/* formula 19 */
E[#<=30;%checks%00] (max:t>1) under Deterrence0

/* formula 20 */
E[#<=30;%checks%] (max:100*interventions/(steps || 1)) under Deterrence0


