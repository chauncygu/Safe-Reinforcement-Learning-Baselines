#=
Helpers to use masking during training with DQN 
Uses a global variable mask that should be defined in main prior to including this file 
=#


function best_action(acts::Vector{A}, val::AbstractArray{T}, problem::M) where {A, T <: Real, M <: Union{POMDP, MDP}}
    best_a = acts[1]
    best_ai = actionindex(problem, best_a) 
    best_val = val[best_ai]
    for a in acts 
        ai = actionindex(problem, a)
        if val[ai] > best_val 
            best_val = val[ai]
            best_ai = ai 
            best_a = a
        end
    end
    return best_a::A
end


# define exploration policy with masking
function masked_linear_epsilon_greedy(max_steps::Int64, eps_fraction::Float64, eps_end::Float64, mask::M) where M <: Union{SafetyMask, JointMask}
    # define function that will be called to select an action in DQN
    # only supports MDP environments
    function action_masked_epsilon_greedy(policy::AbstractNNPolicy, env::POMDPEnvironment, obs, global_step::Int64, rng::AbstractRNG)
        eps = DeepQLearning.update_epsilon(global_step, eps_fraction, eps_end, max_steps)
        acts = safe_actions(pomdp, mask, obs) #XXX using pomdp global variable replace by safe_actions(mask, obs)
        val = actionvalues(policy, obs)
        if rand(rng) < eps
            return (rand(rng, acts), eps)
        else
            return (best_action(acts, val, env.problem), eps)
        end
    end
    return action_masked_epsilon_greedy
end

function masked_evaluation(mask::M) where M <: Union{SafetyMask, JointMask}
    function masked_evaluation_policy(policy::AbstractNNPolicy, env::POMDPEnvironment, n_eval::Int64, max_episode_length::Int64, verbose::Bool)
        avg_r = 0 
        violations = 0
        avg_steps = 0
        for i=1:n_eval
            done = false 
            r_tot = 0.0
            step = 0
            obs = reset(env)
            DeepQLearning.reset!(policy)
            while !done && step <= max_episode_length
                acts = safe_actions(env.problem, mask, obs)
                val = actionvalues(policy, obs)
                act = best_action(acts, val, env.problem)
                obs, rew, done, info = step!(env, act)
                r_tot += rew 
                step += 1
            end
            # println("Episode reward: $r_tot")
            r_tot < 0. ? violations += 1 : nothing
            avg_r += r_tot 
            avg_steps += step
        end
        if verbose
            @printf("Evaluation ... Avg Reward %1.2f | Violations (%%) %2.2f | Avg Steps %3.2f", avg_r/n_eval, violations/n_eval*100, avg_steps/n_eval)
        end
        avg_r /= n_eval
        return  avg_r
    end
    return masked_evaluation_policy
end

struct MaskedNNPolicy{P <: POMDP, N <: AbstractNNPolicy, M <: Union{SafetyMask, JointMask}} <: AbstractNNPolicy
    problem::P
    q::N
    mask::M
end

function POMDPs.action(policy::MaskedNNPolicy, s)
    acts = safe_actions(policy.problem, policy.mask, s)
    val = actionvalues(policy.q, s)
    act = best_action(acts, val, policy.problem)
    @assert in(act, acts)
    return act 
end

function POMDPModelTools.action_info(policy::MaskedNNPolicy, s)
    acts = safe_actions(policy.problem, policy.mask, s)
    val = actionvalues(policy.q, s)
    act = best_action(acts, val, policy.problem)
    @assert in(act, acts)
    probas = compute_probas(policy.problem, policy.mask, s)
    ss = obs_to_scene(policy.problem, s)
    route = get_mdp_state(policy.mask.mdp, policy.problem, ss, PED_ID, CAR_ID).route
    return act, (acts, probas, route)
end

function DeepQLearning.reset!(policy::MaskedNNPolicy)
    return DeepQLearning.reset!(policy.q)
end