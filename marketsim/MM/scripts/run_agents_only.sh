#!/bin/bash

python simMM_example.py --game_name=LadderMM --root_result_folder=./root_result_noMM --num_iteration=10000 --num_background_agents=100 --sim_time=100000.0 --lam=0.005 --lamMM=0.05 --omega=10 --K=10 --n_levels=11 --total_volume=50 --beta_MM=False --inv_driven=False --w0=5 --p=2 --k_min=5 --k_max=20 --max_position=20 --agents_only=True&& \
python simMM_example.py --game_name=LadderMM --root_result_folder=./root_result_noMM --num_iteration=10000 --num_background_agents=100 --sim_time=100000.0 --lam=0.005 --lamMM=0.05 --omega=30 --K=10 --n_levels=11 --total_volume=50 --beta_MM=False --inv_driven=False --w0=5 --p=2 --k_min=5 --k_max=20 --max_position=20 --agents_only=True&& \
python simMM_example.py --game_name=LadderMM --root_result_folder=./root_result_noMM --num_iteration=10000 --num_background_agents=100 --sim_time=100000.0 --lam=0.005 --lamMM=0.05 --omega=60 --K=10 --n_levels=11 --total_volume=50 --beta_MM=False --inv_driven=False --w0=5 --p=2 --k_min=5 --k_max=20 --max_position=20 --agents_only=True&& \
