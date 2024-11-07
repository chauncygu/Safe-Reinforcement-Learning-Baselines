// Train a single strategy, save it, then evaluate it.

/* formula 1 */
strategy PreShielded = minE (total_cost) [#<=30] {} -> {x, t} : <> x>=1 or t>=1

/* formula 2 */
saveStrategy("%resultsdir%/PreShielded.strategy.json", PreShielded)

/* formula 3 */
E[#<=30;%checks%] (max:total_cost) under PreShielded

/* formula 4 */
E[#<=30;%checks%00] (max:t>1) under PreShielded

/* formula 5 */
E[#<=30;%checks%] (max:100*interventions/(steps || 1)) under PreShielded