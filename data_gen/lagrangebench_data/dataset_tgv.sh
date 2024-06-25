#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 nohup ./scripts/dataset_tgv.sh 2>&1 &

DATA_ROOT=/tmp/lagrangebench_data

###### 2D TGV
for seed in {0..199}
do
    echo "Run with seed = $seed"
    python main.py config=cases/tgv.yaml seed=$seed case.dim=2 case.dx=0.02 case.mode=rlx solver.tvf=1.0 case.r0_noise_factor=0.25 io.data_path=$DATA_ROOT/relaxed/
    python main.py config=cases/tgv.yaml seed=$seed case.dim=2 case.dx=0.02 solver.dt=0.0004 solver.t_end=5 case.state0_path=$DATA_ROOT/relaxed/tgv_2_0.02_$seed.h5 io.data_path=$DATA_ROOT/raw/2D_TGV_2500_10kevery100/
done
python gen_dataset.py --src_dir=$DATA_ROOT/raw/2D_TGV_2500_10kevery100/ --dst_dir=$DATA_ROOT/datasets/2D_TGV_2500_10kevery100/ --split=2_1_1

###### 3D TGV
for seed in {0..399}
do
    echo "Run with seed = $seed"
    python main.py config=cases/tgv.yaml seed=$seed case.dim=3 case.dx=0.314159265 case.mode=rlx solver.tvf=1.0 case.r0_noise_factor=0.25 io.data_path=$DATA_ROOT/relaxed/ eos.p_bg_factor=0.01
    python main.py config=cases/tgv.yaml seed=$seed case.dim=3 case.dx=0.314159265 solver.dt=0.005 solver.t_end=30 case.state0_path=$DATA_ROOT/relaxed/tgv_3_0.314159265_$seed.h5 io.data_path=$DATA_ROOT/raw/3D_TGV_8000_10kevery100/ case.viscosity=0.02
done
python gen_dataset.py --src_dir=$DATA_ROOT/raw/3D_TGV_8000_10kevery100/ --dst_dir=$DATA_ROOT/datasets/3D_TGV_8000_10kevery100/ --split=2_1_1
