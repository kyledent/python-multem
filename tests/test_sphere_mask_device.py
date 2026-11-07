#
# test_sphere_mask_device.py
#
# The sphere's slice mask runs as a device functor, like the cuboid's and the cylinder's. It used to be
# a host loop writing one element at a time, which on a device vector is one transfer per pixel: any
# sphere made a GPU render take minutes. The mask must be identical to the host loop's
# (Masker.compute), on the host and on the device, and a sphere region must render in seconds.
#
import os
import time

import numpy as np
import pytest

import multem

N = 512
DR = 1.5


def sphere(centre, radius, n=N, dr=DR):
    m = multem.Masker(n, n, dr)
    m.set_sphere(list(centre), radius)
    return m


def cases():
    rng = np.random.default_rng(4)
    out = []
    for _ in range(12):
        c = rng.uniform(-50, N * DR + 50, 3)
        r = rng.uniform(3, 300)
        out.append((tuple(c), float(r)))
    # surface exactly through pixel centres: (i + 0.5) * DR from the centre
    out.append(((100.0 * DR, 100.0 * DR, 30.0), 20 * DR))
    return out


def slices(c):
    z = c[2]
    return [(z - 400, z - 399), (z - 1.5, z + 1.5), (z + 0.3, z + 3.3), (z + 10, z + 13)]


def test_host_engine_equals_the_loop():
    for c, r in cases():
        m = sphere(c, r)
        for zs, ze in slices(c):
            ref = np.array(m.compute(zs, ze))
            got = np.array(multem.compute_mask_engine(m, zs, ze, False))
            assert np.array_equal(ref, got), (c, r, zs, ze)


@pytest.mark.skipif(not multem.is_gpu_available(), reason="no GPU")
def test_device_engine_equals_the_loop():
    n_true = 0
    for c, r in cases():
        m = sphere(c, r)
        for zs, ze in slices(c):
            ref = np.array(m.compute(zs, ze))
            got = np.array(multem.compute_mask_engine(m, zs, ze, True))
            n_true += int(ref.sum())
            assert np.array_equal(ref, got), (c, r, zs, ze, int((ref != got).sum()))
    assert n_true > 1000


@pytest.mark.skipif(not multem.is_gpu_available(), reason="no GPU")
def test_sphere_region_renders_in_seconds_on_the_gpu():
    os.environ.setdefault("MULTEM_BANDLIMITED_POTENTIAL", "1")
    LZ, DZ = 900.0, 3.0
    inp = multem.Input()
    inp.simulation_type = "EWRS"
    inp.interaction_model = "Multislice"
    inp.potential_type = "Lobato_0_12"
    inp.potential_slicing = "dz_Proj"
    inp.pn_model = "Still_Atom"
    inp.thick_type = "Whole_Spec"
    inp.bwl = False
    inp.E_0 = 300
    inp.nx = inp.ny = N
    inp.spec_lx = inp.spec_ly = N * DR
    inp.spec_lz = LZ
    inp.spec_dz = DZ
    inp.iw_x = [N * DR / 2]
    inp.iw_y = [N * DR / 2]
    inp.spec_atoms = multem.AtomList(
        [(1, 10.0, 10.0, 1.0, 0.0, 1.0, 0, 0), (1, 20.0, 20.0, LZ - 1.0, 0.0, 1.0, 0, 0)])
    ice = multem.IceParameters()
    ice.density = 0.94
    dense = multem.IceParameters()
    dense.density = 1.35
    slab = multem.Masker(N, N, DR)
    slab.set_cuboid((0.0, 0.0, 0.0), (N * DR, N * DR, LZ))
    m = multem.MultiRegionMasker(N, N, DR)
    m.add_region(sphere((N * DR / 2, N * DR / 2, LZ / 2), 300.0), dense)
    m.add_region(sphere((-500.0, -500.0, LZ / 2), 100.0), dense)   # outside the field
    m.add_region(slab, ice, [0, 1])
    m.set_seed(2)
    sc = multem.SystemConfiguration()
    sc.precision = "float"
    sc.device = "device"
    t0 = time.perf_counter()
    out = multem.simulate(sc, inp, m)
    dt = time.perf_counter() - t0
    psi = np.array(out.data[0].psi_coh)
    print(f"\n{N}^2, {int(LZ / DZ)} slices, 2 spheres + slab on the GPU: {dt:.2f} s")
    assert np.isfinite(np.abs(psi)).all()
    assert dt < 30
