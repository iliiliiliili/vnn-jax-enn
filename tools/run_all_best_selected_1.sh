#!/bin/bash

rm screenlog.aenn-*

for input_dim in 10 1 100
do
    for data_ratio in 10 1 100
    do
        # for noise_scale in 0.1 0.01 1
        # do
            echo aenn-id${input_dim}-dr${data_ratio}
            screen -S aenn-id${input_dim}-dr${data_ratio} -L -Logfile screenlog.aenn-id${input_dim}-dr${data_ratio} -dm tools/run_best_selected.sh --input_dim=${input_dim} --data_ratio=${data_ratio} --noise_std=0.1 --noise_std=0.01 --noise_std=1 --agent_id_start=0 --agent_id_end=-1 --agent=vnn_lrelu_init --experiment_group=best_selected_val

            sleep 1
        # done
    done
done