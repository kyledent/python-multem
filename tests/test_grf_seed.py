#
# test_grf_seed.py
#
# The ice's Gaussian random field can be seeded from its masker. With a seed
# the field is the same on every simulation with the same slicing; unseeded
# (the default) it is fresh each time, as before. The measured differences are
# printed so the bounds below can be checked against them (run with -s).
#
import numpy as np
import pytest

import multem

N = 128
DR = 1.0
LZ = 60.0
DZ = 3.0


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
    # two hydrogen anchors at the top and bottom, so that slices exist
    inp.spec_atoms = multem.AtomList(
        [(1, 10.0, 10.0, 1.0, 0.0, 1.0, 0, 0), (1, 20.0, 20.0, LZ - 1.0, 0.0, 1.0, 0, 0)]
    )
    return inp


def _ice():
    ice = multem.IceParameters()
    ice.m1, ice.m2 = 0.0, 1 / 2.88
    ice.s1, ice.s2 = 0.731, 0.081
    ice.a1, ice.a2 = 0.199, 0.801
    ice.density = 0.94
    return ice


def _masker(seed=None):
    masker = multem.Masker(N, N, DR)
    masker.set_ice_parameters(_ice())
    masker.set_cuboid((0.0, 0.0, 0.0), (N * DR, N * DR, LZ))
    if seed is not None:
        masker.set_seed(seed)
    return masker


def _multi_region(seed=None):
    m = multem.MultiRegionMasker(N, N, DR)
    m.add_region(_masker(), _ice())
    if seed is not None:
        m.set_seed(seed)
    return m


def _sc(device):
    sc = multem.SystemConfiguration()
    sc.precision = "float"
    sc.device = device
    return sc


def _potential(device, seed):
    slices = []
    multem.compute_projected_potential(
        _sc(device), _input(), _masker(seed), lambda z0, z1, V: slices.append(np.array(V))
    )
    return np.array(slices)


def _psi(device, masker):
    return np.array(multem.simulate(_sc(device), _input(), masker).data[0].psi_coh)


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def test_default_is_unseeded():
    assert multem.Masker().seed() == -1
    assert multem.MultiRegionMasker().seed() == -1
    m = multem.Masker()
    m.set_seed(12345)
    assert m.seed() == 12345


def test_same_seed_same_field_on_the_host():
    a, b = _potential("host", 7), _potential("host", 7)
    assert a.shape == b.shape and a.shape[0] > 10
    print("host same seed, max rel:", _rel(a, b))
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    a, b = _potential("host", 7), _potential("host", 8)
    da, db = a - a.mean(), b - b.mean()
    corr = float((da * db).sum() / np.sqrt((da * da).sum() * (db * db).sum()))
    print("seeds 7 vs 8, correlation:", corr)
    assert abs(corr) < 0.1


def test_unseeded_still_differs():
    a, b = _potential("host", None), _potential("host", None)
    assert not np.array_equal(a, b)


def test_seeded_exit_wave_on_the_host():
    a, b = _psi("host", _masker(11)), _psi("host", _masker(11))
    assert np.array_equal(a, b)
    a, b = _psi("host", _multi_region(11)), _psi("host", _multi_region(11))
    assert np.array_equal(a, b)
    c = _psi("host", _multi_region(12))
    assert _rel(a, c) > 1e-3


def test_band_limit_spacing_is_compiled_in():
    s = multem.band_limit_spacing(1.5, 1.5)
    assert s > 0
    assert multem.band_limit_spacing(1.0, 2.0) >= multem.band_limit_spacing(1.0, 1.0)


@pytest.mark.skipif(not multem.is_gpu_available(), reason="no GPU")
def test_seeded_on_the_gpu():
    # same seed twice on the GPU: the run-to-run floor, not the 34 % of fresh fields
    a, b = _psi("device", _masker(21)), _psi("device", _masker(21))
    same = _rel(a, b)
    # fresh fields, for scale
    c, d = _psi("device", _masker(None)), _psi("device", _masker(None))
    fresh = _rel(c, d)
    # host against GPU, same seed: the same random field through two FFT paths
    h = _psi("host", _masker(21))
    cross = _rel(a, h)
    print(f"GPU same seed {same:.3g}, GPU unseeded {fresh:.3g}, GPU vs host same seed {cross:.3g}")
    assert same < 1e-4, same
    assert fresh > 100 * max(same, 1e-7), (fresh, same)
    assert cross < 1e-3, cross
