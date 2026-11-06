# python-multem
> Python wrapper for MULTEM

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/rosalindfranklininstitute/python-multem.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/rosalindfranklininstitute/python-multem/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/rosalindfranklininstitute/python-multem.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/rosalindfranklininstitute/python-multem/alerts/)
[![Building](https://github.com/rosalindfranklininstitute/python-multem/actions/workflows/python-package.yml/badge.svg)](https://github.com/rosalindfranklininstitute/python-multem/actions/workflows/python-package.yml)
[![Publishing](https://github.com/rosalindfranklininstitute/python-multem/actions/workflows/python-publish.yml/badge.svg)](https://github.com/rosalindfranklininstitute/python-multem/actions/workflows/python-publish.yml)
[![DOI](https://zenodo.org/badge/201027691.svg)](https://zenodo.org/badge/latestdoi/201027691)

## Installation

To install from the github repository do the following

```sh
export CUDACXX=${PATH_TO_CUDA}/bin/nvcc
python -m pip install git+https://github.com/rosalindfranklininstitute/python-multem.git@master
```

To install from source, clone this repository. The repository has a submodule
for pybind11 so after cloning the repository run

```sh
git submodule update --init --recursive
```

Then do the following:

```sh
export CUDACXX=${PATH_TO_CUDA}/bin/nvcc
python -m pip install .
```

If you would like to run the tests then, clone this repository and then do the following:

```sh
export CUDACXX=${PATH_TO_CUDA}/bin/nvcc
python -m pip install .[test]
```

## Installation for developers

To install for development, clone this repository and then do the following:

```sh
export CUDACXX=${PATH_TO_CUDA}/bin/nvcc
python -m pip install -e .
```

## Performance settings (this fork)

This fork builds against the `performance-optimisation` branch of
[kyledent/MULTEM](https://github.com/kyledent/MULTEM/tree/performance-optimisation) (the `MULTEM`
submodule). Besides its GPU work, that branch moves the host-side preparation of a simulation --
sorting the atoms by z, copying them into place, computing their statistics, identifying the slicing
planes -- onto several CPU threads, and removes work that was repeated. **Every change reproduces the
original results bit for bit**, and each can be checked in place (below).

To install this fork:

```sh
export CUDACXX=${PATH_TO_CUDA}/bin/nvcc
python -m pip install git+https://github.com/kyledent/python-multem.git@multi-region-masker
```

On import, python-multem turns these optimisations on, unless you have set the variable yourself:

| variable | what it does |
|---|---|
| `MULTEM_FAST_SETUP` | compute the atom statistics in one pass instead of about a dozen |
| `MULTEM_SKIP_REDUNDANT_STATS` | the second copy of the atoms reuses the statistics it would recompute |
| `MULTEM_PAR_SORT` | sort the atoms by z on several threads |
| `MULTEM_PAR_COPY` | copy the atoms into place on several threads |
| `MULTEM_REUSE_CONF` | skip rebuilding an atom configuration that is already in place (still atoms) |
| `MULTEM_FAST_PLANES` | skip re-sorting plane positions that are already in order |

Set any of them to `0` before importing `multem` to switch it off.

### Number of threads

By default python-multem uses **75% of the CPUs this process may run on** (at least one) — counted
from the process's affinity mask where the operating system provides one, so a batch slot or a
container limited to some cores is respected. To choose, in order of precedence:

1. `multem.set_num_threads(n)` — at any time; it applies from the next simulation.
2. The environment variable `MULTEM_NUM_THREADS`, set before `multem` is imported — the useful place
   when python-multem is driven by another program, since that program needs no change:

   ```sh
   MULTEM_NUM_THREADS=8 python my_simulation.py
   ```

3. Otherwise `multem.default_num_threads()`, i.e. 75% of `multem.available_cpus()`.

`multem.get_num_threads()` reports the value in effect. More threads is not always faster: these
steps are largely memory-bound, and the measured gains flatten beyond about 8 to 16 threads.
`SystemConfiguration.cpu_nthread` is unrelated — it sets the threads of MULTEM's CPU-only
(`device = "host"`) simulation mode.

### Checking the results

Each optimisation can be run against the original code path in the same process. These are slower —
for verification, not production — and each prints a `[check]` line to stderr counting any
difference (it should be 0): `MULTEM_SORT_CHECK=1`, `MULTEM_COPY_CHECK=1`, `MULTEM_REUSE_CHECK=1`,
`MULTEM_PLANES_CHECK=1`.

## Testing

To run the tests, follow the installation instructions and execute the following:

```sh
pytest
```

## Issues

Please use the [GitHub issue tracker](https://github.com/rosalindfranklininstitute/python-multem/issues) to submit bugs or request features.

## License

Copyright Diamond Light Source, 2019.

Distributed under the terms of the GPLv3 license, python-multem is free and open source software.

