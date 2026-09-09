# Authors

python-multem is a Python interface to MULTEM. It rests on three layers of work,
credited below in the order they were built.

## MULTEM — the simulation engine

**Ivan Lobato** — author of MULTEM (https://github.com/Ivanlh20/MULTEM), the C++/CUDA
multislice package that performs the simulations this project exposes to Python.

The `MULTEM` submodule tracks the Rosalind Franklin Institute's fork of that work
(https://github.com/rosalindfranklininstitute/MULTEM).

Please cite MULTEM's own publications when reporting results produced with this
package; the upstream repository carries the current citation list.

## python-multem — the Python interface

**James Parkhurst** <james.parkhurst@diamond.ac.uk> — author and maintainer of
python-multem (https://github.com/rosalindfranklininstitute/python-multem): the
pybind11 bindings, the build system, the masker and implicit-ice (GRF) model, and
the test suite.

Copyright Diamond Light Source, 2019. Distributed under the terms of the GPLv3.

## This fork

**Kyle Dent** <kyle.dent@gmail.com> — the `multi-region-masker` branch:

- **Multi-region GRF masking** — independent Gaussian Random Field parameters per
  material region, with boolean subtract between regions, so composite geometries
  (a support film surrounding an ice-filled hole) can be represented as GRFs rather
  than as prohibitively many explicit atoms.
- **`SphereMasker`** — a spherical geometric primitive with translation and rotation,
  alongside the existing cuboid and cylinder maskers.
- **X-ray multislice** — `simulate_xray`, a transmission kernel taking per-region
  refractive optical constants (delta, beta) instead of atomic potentials, so the
  same masker geometry can be propagated at X-ray wavelengths.

Developed with **Claude** (Anthropic), credited as co-author on the commits carrying
this work.

## Bundled dependencies

**pybind11** (https://github.com/pybind/pybind11) — Wenzel Jakob and contributors.
