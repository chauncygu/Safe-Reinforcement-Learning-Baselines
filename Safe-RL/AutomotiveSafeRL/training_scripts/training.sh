nohup julia jointmdp_training.jl --log log_nm100 --cost 2 > log100.out &
nohup julia jointmdp_training.jl --log log_nm101 --cost 3 > log101.out &
nohup julia jointmdp_training.jl --log log_nm102 --cost 0.5 > log102.out &
nohup julia jointmdp_training.jl --log log_nm103 --cost 10 > log103.out &
nohup julia jointmdp_training.jl --log log_nm104 --cost 20 > log104.out &

#nohup julia jointmdp_script.jl --log log60 --goal 1 > log60.out &
#nohup julia jointmdp_script.jl --log log61 --goal 1.5 > log61.out &
#nohup julia jointmdp_script.jl --log log62 --goal 2 > log62.out &
#nohup julia jointmdp_script.jl --log log63 --goal 3 > log63.out &
#nohup julia jointmdp_script.jl --log log64 --goal 5 > log64.out &




