nohup julia1.0 train_tracking.jl --seed=1 --entity=car > car1.jodhpur.out &
nohup julia1.0 train_tracking.jl --seed=2 --entity=car > car2.jodhpur.out &
nohup julia1.0 train_tracking.jl --seed=3 --entity=car > car3.jodhpur.out &
nohup julia1.0 train_tracking.jl --seed=4 --entity=car > car4.jodhpur.out &
nohup julia1.0 train_tracking.jl --seed=5 --entity=car > car5.jodhpur.out &

nohup julia1.0 train_tracking.jl --seed=1 --entity=ped > ped1.jodhpur.out &
nohup julia1.0 train_tracking.jl --seed=2 --entity=ped > ped2.jodhpur.out &
nohup julia1.0 train_tracking.jl --seed=3 --entity=ped > ped3.jodhpur.out &
nohup julia1.0 train_tracking.jl --seed=4 --entity=ped > ped4.jodhpur.out &
nohup julia1.0 train_tracking.jl --seed=5 --entity=ped > ped5.jodhpur.out &
