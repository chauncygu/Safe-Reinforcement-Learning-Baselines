// Train a single strategy, save it, then check its safety.

/* formula 1 */
strategy PreShielded = minE (aov) [<=120] {p} -> {t, v}: <> elapsed >= 120

/* formula 2 */
saveStrategy("%resultsdir%/PreShielded.strategy.json", PreShielded)

/* formula 3 */
Pr[<=120] (<>(number_deaths > 0)) under PreShielded