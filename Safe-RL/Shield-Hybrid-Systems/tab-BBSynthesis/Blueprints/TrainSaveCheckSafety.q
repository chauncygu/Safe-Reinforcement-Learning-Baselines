// Train a single strategy, save it, then check its safety.

/* formula 1 */
strategy PreShielded = minE (LearnerPlayer.fired) [<=120] {} -> {p, v}: <> time >= 120

/* formula 2 */
saveStrategy("%resultsdir%/PreShielded.strategy.json", PreShielded)

/* formula 3 */
Pr[<=120] (<> number_deaths > 0) under PreShielded

