// Train a single strategy, save it, then evaluate it.
// HACK: Since this query file is only used for PreShield, haven't implemented a way to count interventions. It will be zero, because I need a number to be printed.

/* formula 1 */
strategy PreShielded = minE (D/1000) [<=120] {} -> {rVelocityEgo, rVelocityFront, rDistance}: <> time >= 120

/* formula 2 */
saveStrategy("%resultsdir%/PreShielded.strategy.json", PreShielded)

/* formula 3 */
E[<=120;%checks%] (max: D/1000)                           under PreShielded

/* formula 4 */
E[<=120;%checks%] (max:(rDistance <= 0))                  under PreShielded

/* formula 5 */
E[<=120;2] (max: 0)


