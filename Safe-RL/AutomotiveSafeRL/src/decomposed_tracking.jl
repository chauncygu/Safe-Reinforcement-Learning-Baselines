
struct SingleAgentTracker <: Updater
    single_pomdp::UrbanPOMDP
    models::Vector{Chain}
    pres_threshold::Float64
    agent_def::VehicleDef
end

struct SingleAgentBelief 
    predictions::Vector{Vector{Float64}}
    obs::Vector{Float64}
    presence::Float64
    single_pomdp::UrbanPOMDP
end

struct MultipleAgentsTracker <: Updater
    pomdp::UrbanPOMDP
    ref_trackers::Dict{Int64, SingleAgentTracker} # bank of uninitialized tracker mapping class to reference tracker
    single_trackers::Dict{Int64, SingleAgentTracker}
end

struct MultipleAgentsBelief
    single_beliefs::Dict{Int64, SingleAgentBelief}
    o::Vector{Float64}
    pomdp::UrbanPOMDP
end


function POMDPs.update(up::MultipleAgentsTracker, bold::MultipleAgentsBelief, a::UrbanAction, o::UrbanObs, verbose=false)
    verbose ? println("Updater keys: ", keys(up.single_trackers)) : nothing
    verbose ? println("belief keys: ", keys(bold.single_beliefs)) : nothing
    ego, car_map, ped_map, obs_map = split_o(o, up.pomdp)
    verbose ? println("cars observed : ", keys(car_map), "ped observed :", keys(ped_map)) : nothing
    bnew = Dict{Int64, SingleAgentBelief}()
    updated_ids = Set{Int64}()
    for (i, veh_map) in enumerate((car_map, ped_map))
        class = i == 1 ? AgentClass.CAR : AgentClass.PEDESTRIAN
        for (vehid, veh) in veh_map # iterate through visible cars
            single_o = vcat(ego, veh, obs_map[1])
            if haskey(bold.single_beliefs, vehid)
                @assert haskey(up.single_trackers, vehid)
                bnew[vehid] = update(up.single_trackers[vehid], bold.single_beliefs[vehid], a, single_o)
                push!(updated_ids, vehid)
            else
                # start new tracker 
                up.single_trackers[vehid] = deepcopy(up.ref_trackers[class])
                verbose ? println("Starting new tracker for vehicle: ", vehid) : nothing
                init_belief = SingleAgentBelief(Vector{Vector{Float64}}(), Vector{Float64}(), 0., up.single_trackers[vehid].single_pomdp)
                bnew[vehid] = update(up.single_trackers[vehid], init_belief, a, single_o)
                push!(updated_ids, vehid)
            end
        end
    end
      
    
    # add absent ped and car for each obstacle
    for (obsid, obs) in obs_map
        absent_obs = vcat(ego, get_normalized_absent_state(up.pomdp, ego), obs)
        for class in (AgentClass.CAR, AgentClass.PEDESTRIAN)
            if class == AgentClass.CAR
                new_id = up.pomdp.max_cars + obsid + 1
            else
                new_id = 100 + up.pomdp.max_peds + obsid + 1
            end
            
            if haskey(up.single_trackers, new_id) && haskey(bold.single_beliefs, new_id)
                b_ = update(up.single_trackers[new_id], bold.single_beliefs[new_id], a, absent_obs)
                if b_.presence > up.single_trackers[new_id].pres_threshold
                    bnew[new_id] = b_
                end
            else
                # start new tracker 
                up.single_trackers[new_id] = deepcopy(up.ref_trackers[class])
                Flux.reset!.(up.single_trackers[new_id].models)
                init_belief = SingleAgentBelief(Vector{Vector{Float64}}(), Vector{Float64}(), 0., up.single_trackers[new_id].single_pomdp)
                b_ = update(up.single_trackers[new_id], init_belief, a, absent_obs)
                if b_.presence > up.single_trackers[new_id].pres_threshold
                    bnew[new_id] = b_
                end
            end
            push!(updated_ids, new_id)
        end
    end
    
    for (oldid, _) in up.single_trackers
        absent_obs = vcat(ego, get_normalized_absent_state(up.pomdp, ego), obs_map[1]) 
        if oldid ∉ updated_ids
            init_belief = SingleAgentBelief(Vector{Vector{Float64}}(), Vector{Float64}(), 0., up.single_trackers[oldid].single_pomdp)
            b_ = update(up.single_trackers[oldid], init_belief, a, absent_obs)
            if b_.presence > up.single_trackers[oldid].pres_threshold
                verbose ? println("Vehicle $oldid disappeared! still tracking") : nothing
                bnew[oldid] = b_
            end
        end
    end
    return MultipleAgentsBelief(bnew, o, up.pomdp)
end

function POMDPs.update(up::SingleAgentTracker, bold::SingleAgentBelief, a::UrbanAction, o::Vector{Float64}) # observation should be consistent with up.pomdp
    n_models = length(up.models)
    predictions = Vector{Vector{Float64}}(undef, n_models)
    presence = 0.
    for (i,m) in enumerate(up.models)
        predictions[i], pres = process_single_entity_prediction(up.single_pomdp, m(o).data, o, up.pres_threshold)
        presence += pres
    end
    presence /= n_models
    return SingleAgentBelief(predictions, o, presence, up.single_pomdp)
end

function BeliefUpdaters.initialize_belief(up::MultipleAgentsTracker, o0::UrbanObs)
    delete!.(Ref(up.single_trackers), k for k in keys(up.single_trackers))
    empty_b = MultipleAgentsBelief(Dict{Int64, SingleAgentBelief}(), Vector{Float64}(), up.pomdp)
    b0 = update(up, empty_b, UrbanAction(0.), o0)
end

function process_single_entity_prediction(pomdp::UrbanPOMDP, b::Vector{<:Real}, o::Vector{<:Real}, pres_threshold::Float64=0.5)
    n_features = pomdp.n_features
    b_ = zeros(3*n_features) # should be 12 (2 vehicles 1 obstacle)
    b_[1:4] = o[1:4] # ego state fully observable
    # get car state from b
    presence = b[5]
    if pres_threshold < presence
        b_[n_features+1:2*n_features] = b[1:4]
    else
        # absent
        b_[n_features+1:2*n_features] = normalized_off_the_grid_pos(pomdp, o[1], o[2])
    end
    b_[2*n_features + 1:end] = o[end - n_features+1:end]
    return b_, presence
end

function create_pedcar_beliefs(pomdp::UrbanPOMDP, b::MultipleAgentsBelief)
    if isempty(b.o)
        return  Dict{NTuple{3, Int64}, PedCarRNNBelief}()
    end
    ego, car_map, ped_map, obs_map = split_o(b.o, pomdp)
    # println("pedestrian detected :", keys(ped_map))
    # println("car detected : ", keys(car_map))
    bkeys = collect(keys(b.single_beliefs))
    car_ids = bkeys[bkeys .< 100]
    if isempty(car_ids)
        push!(car_ids, CAR_ID)
        @assert !haskey(car_map, CAR_ID)
    end
    ped_ids = bkeys[bkeys .> 100]
    if isempty(ped_ids)
        push!(ped_ids, PED_ID)
        @assert !haskey(ped_map, PED_ID)
    end
    absent_ped = get_ped_normalized_absent_state(pomdp, ego)
    absent_car =  get_car_normalized_absent_state(pomdp, ego)
    pedcar_beliefs = Dict{Tuple{Int64, Int64, Int64}, PedCarRNNBelief}()
    obsid = 1 
    for carid in car_ids

        if !haskey(car_map, carid)
            car_o = absent_car
        else
            car_o = car_map[carid]
        end

        for pedid in ped_ids

            if !haskey(ped_map, pedid)
                ped_o = absent_ped
            else
                ped_o = ped_map[pedid]
            end

            if isapprox(car_o, absent_car) && isapprox(ped_o, absent_ped)
                continue
            end          
            o = vcat(ego, car_o, ped_o, obs_map[obsid]) #obstacle does not matter here 
            n_pred, n_features = nothing, nothing
            for present in (carid, pedid)
                if haskey(b.single_beliefs, present)
                    n_pred = length(b.single_beliefs[present].predictions)
                    n_features =  length(b.single_beliefs[present].predictions[1])
                end
            end
            @assert n_pred != nothing
            predictions = Vector{Vector{Float64}}(undef, n_pred)
            for i=1:n_pred
                car_pred = haskey(b.single_beliefs, carid) ? b.single_beliefs[carid].predictions[i][pomdp.n_features+1:2*pomdp.n_features] : car_o
                ped_pred = haskey(b.single_beliefs, pedid) ? b.single_beliefs[pedid].predictions[i][pomdp.n_features+1:2*pomdp.n_features] : ped_o
                predictions[i] = vcat(ego, car_pred, ped_pred) #naïve
            end
            # println("Adding ", carid, " ", pedid, " ", obsid)
            pedcar_beliefs[(carid, pedid, obsid)] = PedCarRNNBelief(predictions, o)
        end
    end                     
    return pedcar_beliefs
end

function get_ped_normalized_absent_state(pomdp, ego)
    ego_x, ego_y, theta, v = ego
    # pos_off =  VecSE2(6.0, 3.0, float(pi)/2)
    pos_off = pomdp.off_grid
    max_ego_dist = get_end(pomdp.env.roadway[pomdp.ego_goal])
    return [pos_off.x/max_ego_dist - ego_x,
                    pos_off.y/max_ego_dist - ego_y,
                    pos_off.θ/float(pi),
                    0. ]
end

function get_car_normalized_absent_state(pomdp, ego)
    ego_x, ego_y, theta, v = ego
    # pos_off =  VecSE2( 26.0, -1.5, 0.)
    pos_off = pomdp.off_grid
    max_ego_dist = get_end(pomdp.env.roadway[pomdp.ego_goal])
    return [pos_off.x/max_ego_dist - ego_x,
                    pos_off.y/max_ego_dist - ego_y,
                    pos_off.θ/float(pi),
                    0. ]
end

function most_likely_scene(pomdp::UrbanPOMDP, b::MultipleAgentsBelief)
    scene = Scene()
    push_ego = false
    ego = get_ego_vehicle(pomdp, b.o)
    push!(scene, ego)
    for (id, sb) in b.single_beliefs
        avg_pred = mean(sb.predictions)
        veh_scene = obs_to_scene(sb.single_pomdp, avg_pred)
        vehind = id > 100 ? findfirst(PED_ID, veh_scene) : findfirst(CAR_ID, veh_scene)
        if vehind != 0
            veh = veh_scene[vehind]
            if id > 100
                push!(scene, Vehicle(veh.state, pomdp.ped_type, id))
            else
                push!(scene, Vehicle(veh.state, pomdp.car_type, id))
            end
        end
    end
    return scene
end

function get_ego_vehicle(pomdp::UrbanPOMDP, o::Vector{Float64})
    obs = deepcopy(o)
    unrescale!(obs, pomdp)
    x,y,θ,v = obs[1:pomdp.n_features]
    ego_state = VehicleState(VecSE2(x,y,θ), pomdp.env.roadway, v)
    ego = Vehicle(ego_state, pomdp.ego_type, EGO_ID)
    return ego
end

struct SingleAgentBeliefOverlay <: SceneOverlay
    b::SingleAgentBelief
    color::Colorant
end

SingleAgentBeliefOverlay(b::SingleAgentBelief;  color = AutoViz.MONOKAY["color4"]) = SingleAgentBeliefOverlay(b, color)

function AutoViz.render!(rendermodel::RenderModel, overlay::SingleAgentBeliefOverlay, scene::Scene, env::OccludedEnv)
    for pred in overlay.b.predictions
        bel = [veh for veh in obs_to_scene(overlay.b.single_pomdp, pred) if veh.id != EGO_ID]
        for veh in bel 
            p = posg(veh.state)
            add_instruction!(rendermodel, render_vehicle, 
                             (p.x, p.y, p.θ, 
                             veh.def.length, veh.def.width, overlay.color,ARGB(1.,1.,1.,0.),ARGB(1.,1.,1.,0.)))
        end
        # AutoViz.render!(rendermodel, GaussianSensorOverlay(sensor=GaussianSensor(), o=bel, color=overlay.color), scene, env) 
    end
end

struct MultipleAgentsBeliefOverlay <: SceneOverlay
    b::MultipleAgentsBelief
    color::Colorant
end

MultipleAgentsBeliefOverlay(b::MultipleAgentsBelief;color = AutoViz.MONOKAY["color4"]) = MultipleAgentsBeliefOverlay(b, color)

function AutoViz.render!(rendermodel::RenderModel, overlay::MultipleAgentsBeliefOverlay, scene::Scene, env::OccludedEnv)
    for (id, bel) in overlay.b.single_beliefs
        bel_overlay = SingleAgentBeliefOverlay(bel, color=overlay.color)
        AutoViz.render!(rendermodel, bel_overlay, scene, env)
    end
end

struct PedCarBeliefOverlay <: SceneOverlay
    pomdp::UrbanPOMDP
    b::PedCarRNNBelief
    color::Colorant
end

function AutoViz.render!(rendermodel::RenderModel, overlay::PedCarBeliefOverlay, scene::Scene, env::OccludedEnv)
    for pred in overlay.b.predictions 
#         bb, _ = process_prediction(overlay.pomdp, pred, overlay.b.obs)
        bel = [veh for veh in obs_to_scene(overlay.pomdp, pred) if veh.id != EGO_ID]
        bel_overlay = GaussianSensorOverlay(sensor=GaussianSensor(), o=bel, color=overlay.color)
        AutoViz.render!(rendermodel, bel_overlay, scene, env)
    end
end
