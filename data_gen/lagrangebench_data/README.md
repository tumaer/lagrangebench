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

To inspect whether the simulated trajectories are of the desired length, we can use
```bash
count_and_check() { find "$1" -type f | sed 's%/[^/]*$%%' | uniq -c | awk -v target="$2" '{if ($1 != target) $1=$1" - Failed!"; n=split($2, a, /[/]/); print a[n]" - "$1}'; }

# and apply it e.g. to TGV with the arguments: 1. dataset directory and 2. target count
count_and_check "/tmp/lagrangebench_data/raw/2D_TGV_2500_10kevery100/" 127
```

## Errata
1. For dam break, one would need to replace lines 182-184 in `jax-sph/case_setup.py` with:
```python
mask, _mask = state["tag"]==Tag.FLUID, _state["tag"]==Tag.FLUID
assert state[k][mask].shape == _state[k][_mask].shape, ValueError(
    f"Shape mismatch for key {k} in state0 file."
)
state[k][mask] = _state[k][_mask]
```