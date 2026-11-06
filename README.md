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

## Band-limited atomic potentials (this fork)

This fork builds against the `band-limited-potential` branch of
[kyledent/MULTEM](https://github.com/kyledent/MULTEM/tree/band-limited-potential) (the `MULTEM`
submodule), which is the `performance-optimisation` branch (next section) plus a correction to how
an atom's potential is put on the grid.

MULTEM samples each atom's projected potential at pixel centres from a radial table that starts one
pixel out from the atom, and extrapolates the table inwards for nearer pixels. At the pixel sizes used
for cryo-EM almost all of an atom's projected potential lies inside that radius, so most of it is
lost: oxygen keeps about 15% of it at 1.5 A/px, under 1% at 3 A/px. The share depends on the element
(oxygen loses more than carbon, hydrogen or nitrogen), so mean inner potentials and the contrast
between materials are both wrong. Binning a simulation afterwards keeps the deficit of the pixel size
it was simulated at.

With `MULTEM_BANDLIMITED_POTENTIAL=1` MULTEM tabulates instead each atom's projected potential
low-pass filtered to the grid -- flat to half the Nyquist frequency, a raised cosine to zero at
Nyquist. A function with nothing at or beyond twice Nyquist sums on the grid to its integral, so
every atom deposits its whole projected potential at any pixel size and any sub-pixel position, and
nothing above Nyquist aliases. The ice model follows suit: its mean is the projected-potential
integral of a water molecule times the number of molecules, with no pixel-size correction, its
variance is the part of its power spectrum that the band passes, and its spectrum carries the same
window.

**python-multem turns this on at import, unless you have set the variable yourself.** It changes
results, by correcting them. Set `MULTEM_BANDLIMITED_POTENTIAL=0` to get MULTEM's original tables and
the ice model's original correction tables back. `multem.band_limited_potential()` reports the setting
in effect. Only neutral atoms are band-limited; ions keep the original tables. Each atom's table
reaches at least 8 pixels, to keep the filter's ringing, so at 1.5 A/px the potential is evaluated at
about 200 pixels per atom instead of 13.

## Performance settings (this fork)

The `performance-optimisation` branch of
[kyledent/MULTEM](https://github.com/kyledent/MULTEM/tree/performance-optimisation), which the
branch above builds on, has GPU work and also moves the host-side preparation of a simulation --
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
| `MULTEM_ATOM_TYPE_CACHE` | build the atom-type coefficient tables once per process for each potential type, grid spacing and band limit, and reuse them in later simulations |

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

