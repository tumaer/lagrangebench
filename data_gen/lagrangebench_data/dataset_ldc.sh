#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 nohup ./scripts/dataset_ldc.sh 2>&1 &

DATA_ROOT=/tmp/lagrangebench_data

##### 2D dataset
python main.py config=cases/ldc.yaml case.dim=2 case.dx=0.02 case.mode=rlx solver.tvf=1.0 case.r0_noise_factor=0.25 io.data_path=$DATA_ROOT/relaxed/ io.p_bg_factor=0.0
python main.py config=cases/ldc.yaml case.dim=2 case.dx=0.02 solver.dt=0.0004 solver.t_end=85 case.state0_path=$DATA_ROOT/relaxed/ldc_2_0.02_123.h5 io.data_path=$DATA_ROOT/raw/2D_LDC_2500_10kevery100/
python gen_dataset.py --src_dir=$DATA_ROOT/raw/2D_LDC_2500_10kevery100/ --dst_dir=$DATA_ROOT/datasets/2D_LDC_2500_10kevery100/ --split=2_1_1 --skip_first_n_frames=1248

# dt_coarse = 0.0004 * 100 = 0.04
# to get 20k samples, we simulate for t = 20k * dt_coarse = 800 (+50 for equilibriation)

##### 3D dataset
python main.py config=cases/ldc.yaml case.dim=3 case.dx=0.041666667 case.mode=rlx solver.tvf=1.0 case.r0_noise_factor=0.25 io.data_path=$DATA_ROOT/relaxed/ io.p_bg_factor=0.0
python main.py config=cases/ldc.yaml case.dim=3 case.dx=0.041666667 solver.dt=0.0009 solver.t_end=1850 case.state0_path=$DATA_ROOT/relaxed/ldc_3_0.041666667_123.h5 io.data_path=$DATA_ROOT/raw/3D_LDC_8160_10kevery100/
python gen_dataset.py --src_dir=$DATA_ROOT/raw/3D_LDC_8160_10kevery100/ --dst_dir=$DATA_ROOT/datasets/3D_LDC_8160_10kevery100/ --split=2_1_1 --skip_first_n_frames=555

# dt_coarse = 0.0009 * 100 = 0.09
# to get 20k samples, we simulate for t = 20k * dt_coarse = 1800 (+50 for equilibriation)
