#!/bin/bash
# run this script with:
# CUDA_VISIBLE_DEVICES=0 nohup ./scripts/dataset_rpf.sh 2>&1 &

DATA_ROOT=/tmp/lagrangebench_data

##### 2D RPF
python main.py config=cases/rpf.yaml case.dim=2 case.dx=0.025 case.mode=rlx solver.tvf=1.0 case.r0_noise_factor=0.25 io.data_path=$DATA_ROOT/relaxed/
python main.py config=cases/rpf.yaml case.dim=2 case.dx=0.025 solver.dt=0.0005 solver.t_end=2050 case.state0_path=$DATA_ROOT/relaxed/rpf_2_0.025_123.h5 io.data_path=$DATA_ROOT/raw/2D_RPF_3200_20kevery100/
python gen_dataset.py --src_dir=$DATA_ROOT/raw/2D_RPF_3200_20kevery100/ --dst_dir=$DATA_ROOT/datasets/2D_RPF_3200_20kevery100/ --split=2_1_1 --skip_first_n_frames=998

###### 3D RPF
python main.py config=cases/rpf.yaml case.dim=3 case.dx=0.05 case.mode=rlx solver.tvf=1.0 case.r0_noise_factor=0.25 io.data_path=$DATA_ROOT/relaxed/
python main.py config=cases/rpf.yaml case.dim=3 case.dx=0.05 solver.dt=0.001 solver.t_end=2050 case.state0_path=$DATA_ROOT/relaxed/rpf_3_0.05_123.h5 io.data_path=$DATA_ROOT/raw/3D_RPF_8000_10kevery100/ eos.p_bg_factor=0.02
python gen_dataset.py --src_dir=$DATA_ROOT/raw/3D_RPF_8000_10kevery100/ --dst_dir=$DATA_ROOT/datasets/3D_RPF_8000_10kevery100/ --split=2_1_1 --skip_first_n_frames=498

# dt_coarse = 0.001 * 100 = 0.1  
# to get 20k samples, we simulate for t = 20k * dt_coarse = 2000 (+50 for equilibriation)

# Also, for a fast particle (vel=1) to cross the box once (length=1) it takes t=1.
# => it takes 10 steps to cross the box with dt_coarse = 0.1
# with 20k samples for train+valid+test, the box is crossed 2000 times