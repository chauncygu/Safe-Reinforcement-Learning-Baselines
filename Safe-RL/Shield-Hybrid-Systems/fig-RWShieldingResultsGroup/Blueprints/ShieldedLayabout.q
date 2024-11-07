//Evaluate the queries with no strategy applied

/* formula 1 */
E[#<=30;%checks%] (max:total_cost)

/* formula 2 */
E[#<=30;%checks%00] (max:t>1)

/* formula 3 */
E[#<=30;%checks%] (max:100*interventions/(steps || 1))
