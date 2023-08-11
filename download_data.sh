#!/bin/bash

declare -A gdrive
gdrive["tgv_2d"]="https://drive.google.com/drive/folders/140_qJ4wwCWryCLv8Dm5syrHjOSlFGj5F"
gdrive["rpf_2d"]="https://drive.google.com/drive/folders/1axqZFRDSlXCsEf_LGz1JRq1rLrOTabmh"
gdrive["ldc_2d"]="https://drive.google.com/drive/folders/153inM2p7pJn27bXs1WaJnfCuY9mW75oY"
gdrive["dam_2d"]="https://drive.google.com/drive/folders/1X-KhmgNNb1mC7acW6WA-vAVZhg4rDPjF"
gdrive["tgv_3d"]="https://drive.google.com/drive/folders/1j20G6AMK47AwHre0QGtGmtW_hKceBX35"
gdrive["rpf_3d"]="https://drive.google.com/drive/folders/1ov9Xds6VSNLSGboht4EasDTRpx2QBFWX"
gdrive["ldc_3d"]="https://drive.google.com/drive/folders/1FjRFjKKuFdjmX5x3Zso5WA4Hk4BjuOHF"

if [ $# -eq 0 ]; then
    echo "Usage: bash download_data.sh <dataset_name>/all"
    exit 1
fi

if [ ! -d "datasets" ]; then
    mkdir datasets
fi

# download the data
if [ $1 == "all" ]; then
    echo "Downloading all datasets"
    for key in ${!gdrive[@]}; do
        echo "Downloading $key"
        gdown --folder --continue  ${gdrive[$key]} -O datasets/
    done
else
    echo "Downloading $1"
    gdown --folder --continue ${gdrive[$1]} -O datasets/
fi
