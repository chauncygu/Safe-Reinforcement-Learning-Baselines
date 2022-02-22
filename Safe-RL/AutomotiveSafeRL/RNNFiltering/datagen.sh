nohup julia1.0 generate_dataset.jl --seed=1 --ntrain=3000 --nval=500 --folder=/scratch/boutonm/ > gen1.jodhpur.out &
nohup julia1.0 generate_dataset.jl --seed=2 --ntrain=3000 --nval=500 --folder=/scratch/boutonm/ > gen2.jodhpur.out &
nohup julia1.0 generate_dataset.jl --seed=3 --ntrain=3000 --nval=500 --folder=/scratch/boutonm/ > gen3.jodhpur.out &
nohup julia1.0 generate_dataset.jl --seed=4 --ntrain=3000 --nval=500 --folder=/scratch/boutonm/ > gen4.jodhpur.out &
nohup julia1.0 generate_dataset.jl --seed=5 --ntrain=3000 --nval=500 --folder=/scratch/boutonm/ > gen5.jodhpur.out &

