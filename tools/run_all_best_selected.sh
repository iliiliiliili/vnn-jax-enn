#!/bin/bash

rm screenlog.aenn-*

for input_dim in 10 1 100
do
    for data_ratio in 10 1 100
    do
        for noise_std in 0.1 0.01 1
        do
            echo aenn-id${input_dim}-dr${data_ratio}-ns${noise_std}
            screen -S aenn-id${input_dim}-dr${data_ratio}-ns${noise_std} -L -Logfile screenlog.aenn-id${input_dim}-dr${data_ratio}-ns${noise_std} -dm tools/run_best_selected.sh --input_dim=${input_dim} --data_ratio=${data_ratio} --noise_std=${noise_std} --agent_id_start=0 --agent_id_end=-1 --agent=all --experiment_group=best_selected_val

            sleep 1
        done
    done
done