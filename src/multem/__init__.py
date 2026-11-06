#
# multem.__init__.py
#
# Copyright (C) 2019 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
# This code is distributed under the GPLv3 license, a copy of
# which is included in the root directory of this package.
#


try:
    from multem._version import version as __version__
except ImportError:
    __version__ = "unknown"


import os
import warnings

import numpy


# Host-side performance settings.
#
# The MULTEM this package builds against can run the host-side preparation of a simulation --
# sorting the atoms by z, copying them into place, their statistics, plane identification -- on
# several CPU threads, and skip work that was repeated. Every one of these reproduces the original
# results bit for bit. MULTEM selects each by an environment variable it reads at call time; this
# package turns them on unless you have set them yourself ("0" switches one off). See README.md.
PERFORMANCE_TOGGLES = (
    "MULTEM_FAST_SETUP",
    "MULTEM_SKIP_REDUNDANT_STATS",
    "MULTEM_PAR_SORT",
    "MULTEM_PAR_COPY",
    "MULTEM_REUSE_CONF",
    "MULTEM_FAST_PLANES",
    "MULTEM_ATOM_TYPE_CACHE",
)

DEFAULT_THREAD_FRACTION = 0.75


def available_cpus():
    """
    The number of CPUs this process may run on: its affinity mask where the operating system
    provides one (so a batch slot or container limited to some cores is respected), otherwise
    the number of CPUs in the machine.

    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def default_num_threads(fraction=DEFAULT_THREAD_FRACTION):
    """
    The default number of host threads: a fraction (75%) of the available CPUs, at least one.

    """
    return max(1, int(available_cpus() * fraction))


def set_num_threads(n):
    """
    Set the number of host threads MULTEM's parallel setup steps use. Takes effect from the next
    simulation.

    Args:
        n (int): The number of threads, at least one

    """
    n = int(n)
    if n < 1:
        raise ValueError("the number of threads must be at least 1, got %d" % n)
    os.environ["MULTEM_NUM_THREADS"] = str(n)


def get_num_threads():
    """
    The number of host threads MULTEM's parallel setup steps will use.

    """
    return int(os.environ["MULTEM_NUM_THREADS"])


def _apply_performance_defaults():
    for name in PERFORMANCE_TOGGLES:
        os.environ.setdefault(name, "1")
    # MULTEM ignores a value that is not a positive integer and falls back to its own default,
    # which would make get_num_threads() report a number that is not in use -- so replace it.
    value = os.environ.get("MULTEM_NUM_THREADS")
    if value is not None:
        try:
            valid = int(value) >= 1
        except ValueError:
            valid = False
        if not valid:
            warnings.warn(
                "MULTEM_NUM_THREADS=%r is not a positive integer; using %d"
                % (value, default_num_threads())
            )
            value = None
    if value is None:
        os.environ["MULTEM_NUM_THREADS"] = str(default_num_threads())


_apply_performance_defaults()


# Band-limited atomic potentials.
#
# MULTEM samples each atom's projected potential at pixel centres from a radial table that starts
# one pixel out from the atom, and extrapolates inside that. At cryo-EM sampling nearly all of the
# potential lies inside, so most of it is lost -- about 85% for oxygen at 1.5 A/px -- in a share
# that depends on the element. With this setting MULTEM tabulates instead the potential
# band-limited to the grid, which an atom deposits in full at any pixel size and sub-pixel
# position, and the ice random field follows suit (no pixel-size correction of its mean; its
# variance the part of its spectrum the band passes). It CHANGES results, by correcting them. This
# package turns it on unless you have set it yourself; "0" restores MULTEM's original tables and
# the ice model's correction tables. See README.md.
BAND_LIMITED_POTENTIAL = "MULTEM_BANDLIMITED_POTENTIAL"

os.environ.setdefault(BAND_LIMITED_POTENTIAL, "1")


def band_limited_potential():
    """
    Whether atomic potentials are band-limited to the grid (MULTEM_BANDLIMITED_POTENTIAL). Read
    by MULTEM at each simulation, so a change takes effect from the next one.

    """
    try:
        return int(os.environ.get(BAND_LIMITED_POTENTIAL, "0")) != 0
    except ValueError:
        return False


from multem_ext import *  # noqa


def slice_spec_atoms(atoms, length_z, num_slices):
    """
    Slice the atoms into a number of subslices

    Args:
        atoms (list): The atom data
        length_z (float): The size of the sample in Z
        num_slices (int): The number of slices to use

    Yields:
        tuple: (z0, lz, atoms)

    """

    # Check the input
    assert length_z > 0, length_z
    assert num_slices > 0, num_slices

    # The slice thickness
    spec_lz = length_z / num_slices

    # Get the atom z
    _, _, _, atom_z, _, _, _, _ = zip(*atoms)
    atom_z = numpy.array(atom_z)
    min_z = numpy.min(atom_z)
    max_z = numpy.max(atom_z)
    assert min_z >= 0
    assert max_z <= length_z
    indices = numpy.arange(0, len(atoms))

    # Loop through the slices
    for i in range(num_slices):
        z0 = i * spec_lz
        z1 = (i + 1) * spec_lz
        if i == num_slices - 1:
            z1 = max(z1, max_z + 1)
        selection = (atom_z >= z0) & (atom_z < z1)
        if numpy.count_nonzero(selection) > 0:
            yield (z0, z1 - z0, AtomList(atoms[i] for i in indices[selection]))
