// Train a single strategy, save it, then evaluate it.

/* formula 1 */
strategy PreShielded = minE (D/1000) [<=120] {} -> {rVelocityEgo, rVelocityFront, rDistance}: <> time >= 120

/* formula 2 */
saveStrategy("%resultsdir%/PreShielded.strategy.json", PreShielded)

/* formula 3 */
Pr[<=120] (<> rDistance <= 0)         under PreShielded

