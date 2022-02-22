using Distributed
addprocs(3)
using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--policy"
        arg_type=String
        default="baseline"
        help = "Choose among 'baseline', 'masked-baseline', 'masked-RL'"
    "--updater"
        arg_type=String 
        default="previous_obs"
        help = "Choose among 'previous_obs', 'tracker'"
    "--scenario"
        arg_type=String 
        default="1-1"
    "--logfile"
        arg_type=String
        default="results.csv"
end
parsed_args = parse_args(ARGS, s)

const N_EPISODES = 1000
const MAX_STEPS = 400

@everywhere begin 
    include("helpers.jl")
end

function parallel_evaluation(pomdp, policy, updater, initialscene_scenario, n_ep=1000, max_steps=400)
    histories = progress_pmap(x->run_sim(pomdp, policy, updater, initialscene_scenario, MersenneTwister(x)), 1:n_ep)
    rewards = zeros(n_ep)
    steps = zeros(n_ep)
    violations = zeros(n_ep)
    for (ep, hist) in enumerate(histories)
        rewards[ep] = discounted_reward(hist)
        steps[ep] = n_steps(hist)
        violations[ep] = is_crash(hist.state_hist[end])
    end
    return rewards, steps, violations
end


pomdp, policy, updater, initialscene_scenario = load_policies_and_environment(parsed_args["policy"], parsed_args["updater"], parsed_args["scenario"])

rewards, steps, violations = parallel_evaluation(pomdp, policy, updater, initialscene_scenario, N_EPISODES, MAX_STEPS)

print_summary(rewards, steps, violations)


## Log output
logfile=parsed_args["logfile"]
df = DataFrame(scenario=parsed_args["scenario"], policy=parsed_args["policy"], updater=parsed_args["updater"],n_episodes = N_EPISODES,
               reward_mean = mean(rewards), reward_std = std(rewards),
               steps_mean = mean(steps), steps_std = std(steps),
               violations_mean = mean(violations), violations_std = std(violations), time=Dates.now())

CSV.write(logfile, df, append = isfile(logfile))
