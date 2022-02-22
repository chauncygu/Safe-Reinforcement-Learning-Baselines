using Flux
using StaticArrays
using ProgressMeter
using POMDPs
using POMDPToolbox
using AutomotiveDrivingModels
using AutomotivePOMDPs
using AutomotiveSensors
using PedCar
using BSON: @load

mdp = PedCarMDP(pos_res=2.0, vel_res=2., ped_birth=0.7, car_birth=0.7)
pomdp = UrbanPOMDP(env=mdp.env,
                    sensor = GaussianSensor(false_positive_rate=0.05, 
                                            pos_noise = LinearNoise(min_noise=0.5, increase_rate=0.05), 
                                            vel_noise = LinearNoise(min_noise=0.5, increase_rate=0.05)),
                   ego_goal = LaneTag(2, 1),
                   max_cars=1, 
                   max_peds=1, 
                   car_birth=0.7, 
                   ped_birth=0.7, 
                   obstacles=false, # no fixed obstacles
                   lidar=false,
                   ego_start=20,
                   ΔT=0.5)

rng = MersenneTwister(1)
policy = RandomPolicy(rng, pomdp, VoidUpdater())


@load "model_1.bson" model
@load "weights_1.bson" weights

@time mean(loss(val_X[i], val_Y[i]) for i=1:length(val_X))

function loss(x, y)
    l = mean(Flux.mse.(model.(x), y))
    truncate!(model)
    reset!(model)
    return l
end

loss.(val_X, val_Y)

xs = Flux.batchseq(val_X)
ys = Flux.batchseq(val_Y)
loss(xs, ys)

