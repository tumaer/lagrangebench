# LagrangeBench dataset generation

To generate the [LagrangeBench datasets](https://zenodo.org/doi/10.5281/zenodo.10021925), we extend the case files provided at https://github.com/tumaer/jax-sph. We first copy the case files and the `main.py` file from JAX-SPH.

```bash
cd data_gen/lagrangebench_data
git clone https://github.com/tumaer/jax-sph.git
cd jax-sph
# We use this specific tag
git checkout v0.0.2

cd ..
cp -r jax-sph/cases/ .
cp jax-sph/main.py .
```

Then we install JAX-SPH
```bash
pip install jax-sph/
# or
# pip install jax-sph==0.0.2

# make sure to have the dasired JAX version, e.g.
pip install jax[cuda12]==0.4.29
```

And the only thing left is running the bash scripts. E.g. for Taylor-Green it is:
```bash
bash dataset_tgv.sh

# cleanup
rm -rf jax-sph/ cases/ main.py
```

Inspect that the simulated trajectories are of the configured length, e.g.
```bash
python count_files.py --src_dir="/tmp/lagrangebench_data/raw/2D_TGV_2500_10kevery100/" --target_count=127
```