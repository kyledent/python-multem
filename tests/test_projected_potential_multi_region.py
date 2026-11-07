#
# test_projected_potential_multi_region.py
#
# compute_projected_potential takes a MultiRegionMasker: the slices carry each region's random field,
# as the exit wave sees them. ice_slice_statistics(ice, dx, dy, dz) returns the (mean, sigma) the
# engine gives one slice of a material, from the same members the simulation calls.
#
import os

import numpy as np
import pytest

import multem

N = 128
DR = 1.5
LZ = 60.0
DZ = 3.0
HOLE_R = 30.0
ANCHORS = [(1, 10.0, 10.0, 1.0, 0.0, 1.0, 0, 0), (1, 20.0, 20.0, LZ - 1.0, 0.0, 1.0, 0, 0)]


@pytest.fixture(autouse=True)
def band_limited():
    saved = os.environ.get("MULTEM_BANDLIMITED_POTENTIAL")
    os.environ["MULTEM_BANDLIMITED_POTENTIAL"] = "1"
    yield
    if saved is None:
        os.environ.pop("MULTEM_BANDLIMITED_POTENTIAL", None)
    else:
        os.environ["MULTEM_BANDLIMITED_POTENTIAL"] = saved


def _input():
    inp = multem.Input()
    inp.simulation_type = "EWRS"
    inp.interaction_model = "Multislice"
    inp.potential_type = "Lobato_0_12"
    inp.potential_slicing = "dz_Proj"
    inp.pn_model = "Still_Atom"
    inp.thick_type = "Whole_Spec"
    inp.bwl = False
    inp.E_0 = 300
    inp.nx = N
    inp.ny = N
    inp.spec_lx = N * DR
    inp.spec_ly = N * DR
    inp.spec_lz = LZ
    inp.spec_dz = DZ
    inp.iw_x = [N * DR / 2]
    inp.iw_y = [N * DR / 2]
    inp.spec_atoms = multem.AtomList(ANCHORS)
    return inp


def _ice(density=0.94):
    p = multem.IceParameters()
    p.density = density
    return p


def _slab():
    m = multem.Masker(N, N, DR)
    m.set_cuboid((0.0, 0.0, 0.0), (N * DR, N * DR, LZ))
    return m


def _hole():
    c = N * DR / 2
    m = multem.Masker(N, N, DR)
    m.set_cylinder([c, c, 0.0], [0, 0, 1], LZ, [HOLE_R], [0], [0])
    return m


def _sc(device):
    sc = multem.SystemConfiguration()
    sc.precision = "float"
    sc.device = device
    return sc


def _slices(masker, device="host"):
    out = []
    multem.compute_projected_potential(
        _sc(device), _input(), masker, lambda z0, z1, V: out.append((z0, z1, np.array(V)))
    )
    return out


def _multi(seed=5):
    m = multem.MultiRegionMasker(N, N, DR)
    m.add_region(_slab(), _ice())
    m.set_seed(seed)
    return m


def _radius():
    y, x = np.mgrid[0:N, 0:N]
    c = N / 2
    return np.hypot(x + 0.5 - c, y + 0.5 - c) * DR


def test_slices_carry_the_region_statistics():
    sl = _slices(_multi())
    assert len(sl) > 10
    worst_mean = worst_sd = 0.0
    for z0, z1, V in sl[1:-1]:  # end slices hold the anchors
        mu, sigma, bl = multem.ice_slice_statistics(_ice(), DR, DR, z1 - z0)
        assert bl is True
        worst_mean = max(worst_mean, abs(V.mean() / mu - 1))
        worst_sd = max(worst_sd, abs(V.std() / sigma - 1))
    print(f"\nslice mean vs ice_slice_statistics: worst rel {worst_mean:.2e}; s.d. worst rel {worst_sd:.2e}")
    assert worst_mean < 1e-4
    assert worst_sd < 1e-3


def test_same_as_a_single_masker():
    single = multem.Masker(N, N, DR)
    single.set_ice_parameters(_ice())
    single.set_cuboid((0.0, 0.0, 0.0), (N * DR, N * DR, LZ))
    single.set_seed(5)
    a = np.array([V for _, _, V in _slices(single)])
    b = np.array([V for _, _, V in _slices(_multi(5))])
    assert a.shape == b.shape
    rel = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
    print(f"\nmulti-region vs single masker, same seed: max rel {rel:.2e}")
    assert rel < 1e-5


def test_vacuum_hole_in_the_slices():
    m = multem.MultiRegionMasker(N, N, DR)
    m.add_region(_hole(), _ice(0.0))
    m.add_region(_slab(), _ice(), [0])
    m.set_seed(3)
    r = _radius()
    inside, outside = r < HOLE_R - 3 * DR, r > HOLE_R + 3 * DR
    # a z-axis cylinder's z extent is taken as z +/- r, so the sentinels widen the slicing beyond the
    # slab; only slices inside the slab hold ice
    inner = [(z0, z1, V) for z0, z1, V in _slices(m) if z0 >= 1.0 and z1 <= LZ - 1.0]
    assert len(inner) > 5
    for z0, z1, V in inner:
        mu = multem.ice_slice_statistics(_ice(), DR, DR, z1 - z0)[0]
        assert abs(float(V[inside].mean())) < 1e-3 * mu
        assert abs(float(V[outside].mean()) / mu - 1) < 0.1


def test_statistics_follow_the_overrides():
    a = multem.ice_slice_statistics(_ice(), DR, DR, 3.0)
    p = _ice()
    p.m0, p.v0 = 20.0, 4 * 10195.82
    b = multem.ice_slice_statistics(p, DR, DR, 3.0)
    assert b[1] / a[1] == pytest.approx(2.0, rel=1e-12)
    c = _ice()
    c.m0 = 10.0
    assert b[0] / multem.ice_slice_statistics(c, DR, DR, 3.0)[0] == pytest.approx(2.0, rel=1e-12)
    assert multem.ice_slice_statistics(_ice(), DR, DR, 6.0)[0] / a[0] == pytest.approx(2.0, rel=1e-12)
    os.environ["MULTEM_BANDLIMITED_POTENTIAL"] = "0"
    legacy = multem.ice_slice_statistics(_ice(), DR, DR, 3.0)
    assert legacy[2] is False and legacy[0] != a[0]


@pytest.mark.skipif(not multem.is_gpu_available(), reason="no GPU")
def test_gpu_agrees_with_host():
    a = np.array([V for _, _, V in _slices(_multi(), "device")])
    b = np.array([V for _, _, V in _slices(_multi(), "host")])
    rel = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    print(f"\nprojected potential, multi-region, GPU vs host: max rel {rel:.2e}")
    assert rel < 1e-4
