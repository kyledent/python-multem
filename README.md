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
nothing above Nyquist aliases. As tabulated, the sum matches the integral to about 1e-4: one atom's grid
sum varies by about 1e-4 with its sub-pixel position, independent of slice thickness and box size. The
ice model's mean is computed analytically, not summed from the grid. The ice model follows suit: its mean is the projected-potential
integral of a water molecule times the number of molecules, with no pixel-size correction, its
variance is the part of its power spectrum that the band passes, and its spectrum carries the same
window.

**python-multem turns this on at import, unless you have set the variable yourself.** It changes
results, by correcting them. Set `MULTEM_BANDLIMITED_POTENTIAL=0` to get MULTEM's original tables and
the ice model's original correction tables back. `multem.band_limited_potential()` reports the setting
in effect. Only neutral atoms are band-limited; ions keep the original tables. Each atom's table
reaches at least 8 pixels, to keep the filter's ringing, so at 1.5 A/px the potential is evaluated at
about 200 pixels per atom instead of 13.

## Ice and material regions (this fork)

A `Masker` describes one region of random-field ice; a `MultiRegionMasker` several, each with its own
material (`IceParameters`) and regions it subtracts.

- **Grid.** Masks sample the same grid points as the atom potential, pixel i at x = i * dx (its corner).
  Before, masks sampled pixel centres, so regions sat half a pixel from the atoms.
- **Partial volume.** By default a slice takes a region wholly or not at all, by where the slice's
  midpoint falls, so at normal incidence a region's path is a whole number of slices: off by up to a
  slice per pair of faces. `set_partial_volume(n)` weights each pixel by the fraction of the slice inside
  the region instead. For spheres and cuboids (at any rotation) the fraction is exact, from where the ray
  meets the shape, and costs one evaluation per slice; cylinders take n sub-slice samples (error at most
  dz/(2n) per face). The mean scales with the fraction and the texture's variance with it too.
- **Spectrum.** `IceParameters.set_spectrum_table(q, P)` replaces the two-Gaussian power spectrum with a
  table (it needs the band-limited potential). `v0` is then the variance per molecule over the table's own
  range.
- **Convex regions.** `Masker.set_convex(A, centre, planes, has_inner, inner_A, inner_centre, inner_planes)`
  takes an optional quadric (p − c)ᵀA(p − c) ≤ 1 (an ellipsoid, or an elliptic cylinder when A is singular, at any
  orientation) intersected with up to 16 half-spaces n·p ≤ d. It can subtract an inner set given the same way.
  Ellipsoids with cuts, boxes at any rotation, slabs with a rectangle or disc footprint, and ellipsoidal shells all
  take this form. A ray along z meets such a set in one interval, so partial-volume fractions are exact here too.
  The masker's rotation and translation apply as for the other shapes. Faces are closed for planes (n·p ≤ d) and half-open
  in z for the interval; the cuboid shape is half-open on every face. So a box built from planes and the cuboid
  disagree only for a pixel column lying exactly in a vertical face, a set of measure zero that a synthetic
  scene on a lattice can still hit.
- **Vacuum.** A region with density 0 adds nothing, and other regions can still subtract its shape.
- **Test hooks.** `compute_mask_engine`, `compute_fraction_engine` and `compute_region_labels_engine` return what
  the simulation computes for one slice. The last gives each pixel's owning region after subtractions: its index,
  −1 for none, or −2 for an overlap.
- **Statistics.** `ice_slice_statistics(ice, dx, dy, dz)` returns the mean and sigma the engine gives one
  slice of a material, and whether the band-limited potential was in force.

Two behaviours inherited from upstream are worth knowing:

- The ice is added at a third of its value wherever the atoms' potential is positive
  (`AddRandomFieldAndPotential`), so atoms inside a region take away part of the ice under them.
- The engine adds two hydrogen atoms at (0, 0, zmin) and (0, 0, zmax) of the masker to fix the slicing
  range. Together with the previous point, that lowers the ice near the grid's corner in the slices at a
  region's faces, so full-grid means of those slices include it. A render that crops its margin does not
  keep those pixels.

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

