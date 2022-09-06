
mkdir -p ../results/single_runs
ln -s ../results
ln -s ../results/single_runs

python ./enn/experiments/neurips_2021/run_testbed_best_selected.py --input_dim=10 --data_ratio=1 --noise_std=0.1 --agent_id_start=10 --agent_id_end=11 --agent=vnn_selected --seed=2605