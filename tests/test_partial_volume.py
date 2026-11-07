#
# test_partial_volume.py
#
# With set_partial_volume(n), each slice pixel is weighted by the fraction of the slice's depth inside
# the region (n sub-slice midpoint tests) instead of the 0/1 test at the slice midpoint. At normal
# incidence the 0/1 test counts a region as a whole number of slices, so its path is off by up to
# a slice per face pair; with fractions the error is at most dz/(2n) per face.
#
import os

import numpy as np
import pytest

import multem

N = 64
DR = 1.5


@pytest.fixture(autouse=True)
def band_limited():
    saved = os.environ.get("MULTEM_BANDLIMITED_POTENTIAL")
    os.environ["MULTEM_BANDLIMITED_POTENTIAL"] = "1"
    yield
    if saved is None:
        os.environ.pop("MULTEM_BANDLIMITED_POTENTIAL", None)
    else:
        os.environ["MULTEM_BANDLIMITED_POTENTIAL"] = saved


def devices():
    return ["host"] + (["device"] if multem.is_gpu_available() else [])


def slab_masker(z0, thickness):
    m = multem.Masker(N, N, DR)
    m.set_cuboid((0.0, 0.0, z0), (N * DR, N * DR, thickness))
    return m


def test_default_is_off():
    assert multem.Masker().partial_volume() == 0
    assert multem.MultiRegionMasker().partial_volume() == 0
    m = multem.Masker()
    m.set_partial_volume(16)
    assert m.partial_volume() == 16


@pytest.mark.parametrize("device", devices())
def test_fraction_of_a_face(device):
    n = 32
    rng = np.random.default_rng(1)
    for _ in range(20):
        z0 = rng.uniform(0, 10)
        m = slab_masker(z0, 50.0)
        zs = rng.uniform(z0 - 2, z0 + 2)
        ze = zs + 3.0
        exact = np.clip((ze - max(zs, z0)) / (ze - zs), 0, 1)
        f = np.array(multem.compute_fraction_engine(m, zs, ze, n, device == "device"))
        assert np.all(f == f[0])
        assert abs(f[0] - exact) <= 0.5 / n + 1e-12, (zs, z0, f[0], exact)


def test_device_fraction_equals_host():
    if not multem.is_gpu_available():
        pytest.skip("no GPU")
    m = multem.Masker(N, N, DR)
    m.set_sphere([N * DR / 2 + 0.3, N * DR / 2 - 0.2, 20.0], 25.0)
    for zs in np.arange(-10, 50, 2.7):
        h = np.array(multem.compute_fraction_engine(m, zs, zs + 3.0, 16, False))
        d = np.array(multem.compute_fraction_engine(m, zs, zs + 3.0, 16, True))
        assert np.array_equal(h, d), zs


def _input(LZ, DZ):
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
    # anchors well outside the slab in z, in a corner, so they fix the slicing and not the slab
    inp.spec_atoms = multem.AtomList([(1, 1.0, 1.0, 0.0, 0.0, 1.0, 0, 0), (1, 1.0, 1.0, LZ, 0.0, 1.0, 0, 0)])
    return inp


def effective_thickness(z0, thickness, dz, samples, device="host"):
    """The slab's projected mean, as an ice path in A: sum over slices of the slice's mean potential,
    divided by the potential per A of path (from ice_slice_statistics). The mean is taken over the
    middle of the grid: the engine puts two hydrogen sentinels at (0, 0, zmin) and (0, 0, zmax) of the
    masker, here the slab's faces, and ice where the atomic potential is positive is counted at a
    third, so the corner of the face slices holds less. A negligible variance (v0) removes the texture
    from that local mean."""
    LZ = 120.0
    ice = multem.IceParameters()
    ice.density = 0.94
    ice.v0 = 1e-9
    m = multem.MultiRegionMasker(N, N, DR)
    m.add_region(slab_masker(z0, thickness), ice)
    m.set_seed(1)
    m.set_partial_volume(samples)
    sc = multem.SystemConfiguration()
    sc.precision = "double"
    sc.device = device
    total = []
    multem.compute_projected_potential(sc, _input(LZ, dz), m, lambda a, b, V: total.append((a, b, np.array(V))))
    mu_per_A = multem.ice_slice_statistics(ice, DR, DR, 1.0)[0]
    # the anchors' potential is the same with and without ice; take it off with a vacuum twin
    v = multem.MultiRegionMasker(N, N, DR)
    vac = multem.IceParameters()
    vac.density = 0.0
    v.add_region(slab_masker(z0, thickness), vac)
    v.set_partial_volume(samples)
    base = []
    multem.compute_projected_potential(sc, _input(LZ, dz), v, lambda a, b, V: base.append(np.array(V)))
    mid = (slice(N // 4, 3 * N // 4), slice(N // 4, 3 * N // 4))
    s = sum(float((V - B)[mid].mean()) for (_, _, V), B in zip(total, base))
    return s / mu_per_A


@pytest.mark.parametrize("dz", [1.5, 3.0, 6.0])
def test_thin_slab_path_at_sub_slice_offsets(dz):
    worst_off = worst_on = 0.0
    n = 32
    for k in range(6):
        z0 = 40.0 + k * dz / 6
        off = effective_thickness(z0, 20.0, dz, 0) - 20.0
        on = effective_thickness(z0, 20.0, dz, n) - 20.0
        worst_off, worst_on = max(worst_off, abs(off)), max(worst_on, abs(on))
    print(f"\ndz {dz}: 20 A slab, worst path error midpoint {worst_off:.3f} A, partial volume ({n}) {worst_on:.4f} A")
    assert worst_on <= dz / n + 1e-6
    assert worst_off > 4 * worst_on or worst_off < 1e-6


@pytest.mark.skipif(not multem.is_gpu_available(), reason="no GPU")
def test_thin_slab_on_the_gpu():
    h = effective_thickness(41.3, 20.0, 3.0, 32, "host")
    d = effective_thickness(41.3, 20.0, 3.0, 32, "device")
    print(f"\n20 A slab, partial volume: host {h:.5f} A, GPU {d:.5f} A")
    assert abs(h - 20.0) <= 3.0 / 32 + 1e-6
    assert abs(d - h) < 1e-4


def test_vacuum_child_is_taken_off():
    ice = multem.IceParameters()
    ice.density = 0.94
    vac = multem.IceParameters()
    vac.density = 0.0
    hole = multem.Masker(N, N, DR)
    hole.set_cuboid((20 * DR, 20 * DR, 0.0), (10 * DR, 10 * DR, 200.0))
    m = multem.MultiRegionMasker(N, N, DR)
    m.add_region(hole, vac)
    m.add_region(slab_masker(41.3, 20.0), ice, [0])
    m.set_seed(1)
    m.set_partial_volume(16)
    sc = multem.SystemConfiguration()
    sc.precision = "double"
    sc.device = "host"
    V = []
    multem.compute_projected_potential(sc, _input(120.0, 3.0), m, lambda a, b, v: V.append(np.array(v)))
    P = np.sum(V, axis=0)
    inside = P[22:28, 22:28]
    assert float(np.abs(inside).max()) < 1e-9
    assert float(P[5:15, 40:60].mean()) > 1.0
