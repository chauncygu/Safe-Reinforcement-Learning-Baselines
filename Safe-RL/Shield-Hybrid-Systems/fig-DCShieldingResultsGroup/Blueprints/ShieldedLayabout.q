//Evaluate the queries with no strategy applied

/* formula 2 */
E[<=120;%checks%] (max:Monitor.dist + switches*1.0)

/* formula 3 */
E[<=120;%checks%] (max:number_deaths > 0)

/* formula 4 */
E[<=120;%checks%] (max:interventions)
