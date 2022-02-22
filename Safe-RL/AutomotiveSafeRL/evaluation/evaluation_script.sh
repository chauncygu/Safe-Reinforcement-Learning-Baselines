julia evaluation.jl --policy=baseline --updater=previous_obs --scenario=1-2 --logfile=results3.csv
julia evaluation.jl --policy=baseline --updater=tracker --scenario=1-2 --logfile=results3.csv
julia evaluation.jl --policy=masked-baseline --updater=tracker --scenario=1-2 --logfile=results3.csv
julia evaluation.jl --policy=masked-RL --updater=tracker --scenario=1-2 --logfile=results3.csv

julia evaluation.jl --policy=baseline --updater=previous_obs --scenario=1-1 --logfile=results3.csv
julia evaluation.jl --policy=baseline --updater=tracker --scenario=1-1 --logfile=results3.csv
julia evaluation.jl --policy=masked-baseline --updater=tracker --scenario=1-1 --logfile=results3.csv
julia evaluation.jl --policy=masked-RL --updater=tracker --scenario=1-1 --logfile=results3.csv

julia evaluation.jl --policy=baseline --updater=previous_obs --scenario=2-1 --logfile=results3.csv
julia evaluation.jl --policy=baseline --updater=tracker --scenario=2-1 --logfile=results3.csv
julia evaluation.jl --policy=masked-baseline --updater=tracker --scenario=2-1 --logfile=results3.csv
julia evaluation.jl --policy=masked-RL --updater=tracker --scenario=2-1 --logfile=results3.csv

julia evaluation.jl --policy=baseline --updater=previous_obs --scenario=2-2 --logfile=results3.csv
julia evaluation.jl --policy=baseline --updater=tracker --scenario=2-2 --logfile=results3.csv
julia evaluation.jl --policy=masked-baseline --updater=tracker --scenario=2-2 --logfile=results3.csv
julia evaluation.jl --policy=masked-RL --updater=tracker --scenario=2-2 --logfile=results3.csv

julia evaluation.jl --policy=baseline --updater=previous_obs --scenario=3 --logfile=results3.csv
julia evaluation.jl --policy=baseline --updater=tracker --scenario=3 --logfile=results3.csv
julia evaluation.jl --policy=masked-baseline --updater=tracker --scenario=3 --logfile=results3.csv
julia evaluation.jl --policy=masked-RL --updater=tracker --scenario=3 --logfile=results3.csv