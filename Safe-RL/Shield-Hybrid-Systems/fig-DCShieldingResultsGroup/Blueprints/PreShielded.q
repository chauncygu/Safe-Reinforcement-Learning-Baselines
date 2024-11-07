// Train a single strategy, save it, then evaluate it.

/* formula 1 */
strategy PreShielded = minE(Monitor.dist + switches*1.0) [<=120] {Converter.location} -> {x1, x2}: <> time >= 120

/* formula 2 */
saveStrategy("%resultsdir%/PreShielded.strategy.json", PreShielded)

/* formula 3 */
E[<=120;%checks%] (max:Monitor.dist + switches*1.0) under PreShielded

/* formula 4 */
E[<=120;%checks%] (max:number_deaths > 0) under PreShielded

/* formula 5 */
E[<=120;%checks%] (max:interventions) under PreShielded