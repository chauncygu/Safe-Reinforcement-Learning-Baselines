function evaluate_safety(mechanics, policy, number_of_runs;
        run_duration=120,
        min_v_on_impact=1,
        unlucky=false)
    
    safety_violations_observed = 0
    unsafe_trace = []
    rand_step = eps()

    for run in 1:number_of_runs
        v, p = 0, rand(7:rand_step:10)
        # Simulate the ball for run_duration seconds
        vs, ps, ts = simulate_sequence(mechanics, v, p, policy, run_duration,
            min_v_on_impact=min_v_on_impact,
            unlucky=unlucky)
        # See if it ends at v=0, p=0
        if last(vs) == 0 && last(ps) == 0
            safety_violations_observed += 1
        end
    end
    (; safety_violations_observed, number_of_runs)
end

# It does not choose a random policy. It returns a policy that acts randomly.
function random_policy(hit_chance)
	return (v, p) -> 
		if rand(0:eps():1) <= hit_chance
			"hit"
		else
			"nohit"
		end
end
