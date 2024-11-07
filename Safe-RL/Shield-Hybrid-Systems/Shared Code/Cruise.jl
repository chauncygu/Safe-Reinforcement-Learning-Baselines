#=
	using StatsBase
	using Unzip
=#

# Define actions as an enums. The int map of the enum is also its acceleration factor.
@enum CCAction backwards=-2 neutral=0 forwards=2

begin 
	import Base.+
	
	a::Number + b::CCAction = a+Int(b)
	
	a::CCAction + b::Number = CCAction(Int(a) + b)
end

struct CCMechanics
	t_act::Number # Period between actions
	ego_sensor_range::Number # front is invisible after this
	v_ego_min::Number
	v_ego_max::Number
	v_front_min::Number
	v_front_max::Number
end

ccmechanics = CCMechanics(1, 200, -10, 20, -8, 20)

function random_front_behaviour(mechanics::CCMechanics, s)
    # Check if inside sensor range.
    if s[3] <= mechanics.ego_sensor_range
        return sample([backwards neutral forwards],
            Weights([1/3 for i in 1:3]))
    else
        if rand() < 0.5 && s[1] > mechanics.v_front_min
            return rand(mechanics.v_front_min - 1:1:s[1])
        else
            # Stays outside range
            return mechanics.v_front_max 
        end
    end
end

function get_random_front_behaviour(mechanics)
	s -> random_front_behaviour(mechanics, s)
end

function speed_limit(min, max, v, action::CCAction)
	if action == backwards && v <= min
		return neutral
	elseif action == forwards && v >= max
		return neutral
	else
		return action
	end
end

function throw_if_invalid_front_action(mechanics, s, front_action)
	if s[3] <= mechanics.ego_sensor_range && typeof(front_action) != CCAction
		
		throw(ArgumentError("""Invalid action for front. 
		Front chose an action that is not appropriate inside of sensor range. 
		Action chosen: $front_action
		State: $s"""))

	end
	if s[3] > mechanics.ego_sensor_range && typeof(front_action) == CCAction
		
		throw(ArgumentError("""Invalid action for front. 
		Front chose an action that is only appropriate inside of sensor range. 
		Action chosen: $front_action
		State: $s"""))
		
	end
end


function simulate_point(mechanics::CCMechanics, 
        front_behaviour::Function, 
        s, 
        action::CCAction)
    
    front_action = front_behaviour(s)
    simulate_point(mechanics, front_action, s, action)
end

function simulate_point(mechanics::CCMechanics, 
        front_action::Union{CCAction, Number}, 
        s, 
        action::CCAction)

    v_ego, v_front, distance = s

    old_vel = v_front - v_ego;

    throw_if_invalid_front_action(mechanics, s, front_action)
    
    # Update v_front. 
    # Front behaviour varies depending on whether it is inside sensor range.
    if distance <= mechanics.ego_sensor_range

        front_action′ = speed_limit(mechanics.v_front_min, 
            mechanics.v_front_max, 
            v_front,
            front_action)
        
        v_front = v_front + front_action′
    else 
        # front can choose to come back into sensor range 
        # at a velocity less than the ego
        if front_action < v_ego

            # Just for good measure
            v_front = clamp(front_action,
                mechanics.v_front_min - 1, mechanics.v_front_max + 1)

            # Need to update old_vel. 
            # This update happens before the call to updateDiscrete() 
            # in the UPPAAL model.
            old_vel = v_front - v_ego;
            distance = 200
        else
            v_front = 0
        end
    end

    action′ = speed_limit(mechanics.v_ego_min, 
        mechanics.v_ego_max, 
        v_ego,
        action)
    
    v_ego = v_ego + action′

    new_vel = v_front - v_ego;

    distance += (old_vel + new_vel)/2;
    distance = min(distance, mechanics.ego_sensor_range + 1)
    
    (v_ego, v_front, distance)
end
    

function simulate_sequence(mechanics::CCMechanics, front_behaviour::Function, duration, s0, policy::Function)
    states, times = [s0], [0.0]
    s, t = s0, 0
    while times[end] <= duration - mechanics.t_act
        action = policy(s)
        s = simulate_point(mechanics, front_behaviour, s, action)
		t += mechanics.t_act
        push!(states, s)
        push!(times, t)
    end
    (;states, times)
end

function plot_sequence(states, times; dim=1, plotargs...)
	unzipped = unzip(states)
	layout = (2, 1)
	linewidth = 4
	
	p1 = plot(times, unzipped[1]; 
		label="v_ego",
		ylabel="v",
		xlabel="t", 
		linewidth=linewidth,
		linecolor=:green,
		plotargs...)
	
	plot!(times, unzipped[2];
		linewidth=linewidth,
		linecolor=:blue,
		label="v_front")

	p2 = plot(times, unzipped[3]; 
		xlabel="t",
		ylabel="d",
		label="distance",
		linewidth=linewidth,
		linecolor=:red,
		plotargs...)
	
	plot(p1, p2, layout=layout, size=(800, 400))
end