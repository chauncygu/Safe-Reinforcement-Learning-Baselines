// Train a single strategy, save it, then evaluate it.

/* formula 1 */
strategy PreShielded = minE (total_cost) [#<=30] {} -> {x, t} : <> x>=1 or t>=1

/* formula 2 */
saveStrategy("%resultsdir%/PreShielded.strategy.json", PreShielded)

/* formula 3 */
Pr[#<=30] (<> t>1) under PreShielded