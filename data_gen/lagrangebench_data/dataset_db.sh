#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 nohup ./scripts/dataset_db.sh >> db_dataset.out 2>&1 &

DATA_ROOT=/tmp/lagrangebench_data

##### 2D dataset
for seed in {0..114}  # 15 trajectories blew up and were discarded
do
    echo "Run with seed = $seed"
    python main.py config=cases/db.yaml seed=$seed case.mode=rlx solver.tvf=1.0 case.r0_noise_factor=0.25 io.data_path=$DATA_ROOT/relaxed/
    python main.py config=cases/db.yaml seed=$seed case.state0_path=$DATA_ROOT/relaxed/db_2_0.02_$seed.h5 io.data_path=$DATA_ROOT/raw/2D_DB_5740_20kevery100/
done
# 15 blowing up runs were removed from the dataset, and 100 were kept
# use `count_files_db.py` to detect defect runs
python gen_dataset.py --src_dir=$DATA_ROOT/raw/2D_DB_5740_20kevery100/ --dst_dir=$DATA_ROOT/datasets/2D_DB_5740_20kevery100/ --split=2_1_1


### Number of particles
# 100x50=5000 water particles
# 106x274 outer box, i.e. 106x274 - 100x268 = 2244 wall particles
# with only one wall layer, wall particles are 2*(100+270) = 740
# => 7244 particles with SPH and 5740 for dataset

### Number of seeds for a given number of training samples 
# t_end = 12
# dt_coarse = 0.0003*100 = 0.03
# num_samples = 12/0.03 = 400  (+1 for initial state)
# => for 40k train+valid+test samples, we need 40k/400 = 100 seeds
