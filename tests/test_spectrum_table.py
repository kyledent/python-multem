#
# test_spectrum_table.py
#
# IceParameters.set_spectrum_table(q, P) replaces the two-Gaussian power spectrum of the ice's
# random field with a tabulated one (linear between knots, P(q0) below the first knot, 0 above the
# last). A table sampled densely from the two Gaussians must give the same ice; P's scale must not
# matter (each slice is normalised to sigma); the default (no table) is unchanged; a table needs the
# band-limited potential. The single-region masker now also honours IceParameters.v0.
#
import os

import numpy as np
import pytest

import multem

N = 128
DR = 1.5
LZ = 60.0
DZ = 3.0
ANCHORS = [(1, 10.0, 10.0, 1.0, 0.0, 1.0, 0, 0), (1, 20.0, 20.0, LZ - 1.0, 0.0, 1.0, 0, 0)]
GAUSS = dict(a1=0.199, a2=0.801, m1=0.0, m2=1 / 2.88, s1=0.731, s2=0.081)


@pytest.fixture(autouse=True)
def band_limited():
    saved = os.environ.get("MULTEM_BANDLIMITED_POTENTIAL")
    os.environ["MULTEM_BANDLIMITED_POTENTIAL"] = "1"
    yield
    if saved is None:
        os.environ.pop("MULTEM_BANDLIMITED_POTENTIAL", None)
    else:
        os.environ["MULTEM_BANDLIMITED_POTENTIAL"] = saved


def two_gaussians(q):
    g = GAUSS
    return g["a1"] * np.exp(-0.5 * (q - g["m1"]) ** 2 / g["s1"] ** 2) + g["a2"] * np.exp(
        -0.5 * (q - g["m2"]) ** 2 / g["s2"] ** 2
    )


def dense_table(scale=1.0):
    q = np.linspace(1e-4, 4.0, 40001)
    return q, scale * two_gaussians(q)


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


def _ice(table=None, v0=-1.0):
    p = multem.IceParameters()
    for k, v in GAUSS.items():
        setattr(p, k, v)
    p.density = 0.94
    p.v0 = v0
    if table is not None:
        p.set_spectrum_table(list(table[0]), list(table[1]))
    return p


def _masker(ice, seed=7):
    m = multem.Masker(N, N, DR)
    m.set_ice_parameters(ice)
    m.set_cuboid((0.0, 0.0, 0.0), (N * DR, N * DR, LZ))
    m.set_seed(seed)
    return m


def _sc(device):
    sc = multem.SystemConfiguration()
    sc.precision = "float"
    sc.device = device
    return sc


def _slices(masker, device="host"):
    out = []
    multem.compute_projected_potential(
        _sc(device), _input(), masker, lambda z0, z1, V: out.append(np.array(V))
    )
    return np.array(out)


def _psi(masker, device="host"):
    return np.array(multem.simulate(_sc(device), _input(), masker).data[0].psi_coh)


def test_validation():
    p = multem.IceParameters()
    assert not p.has_spectrum_table()
    assert p.spectrum_q == [] and p.spectrum_P == []
    for q, P in [
        ([0.1], [1.0]),                    # one knot
        ([0.1, 0.2], [1.0]),               # lengths differ
        ([0.0, 0.2], [1.0, 1.0]),          # q0 not > 0
        ([0.2, 0.1], [1.0, 1.0]),          # not increasing
        ([0.1, 0.1], [1.0, 1.0]),          # not strictly increasing
        ([0.1, 0.2], [1.0, -1.0]),         # negative power
        ([0.1, 0.2], [0.0, 0.0]),          # zero everywhere
        ([0.1, float("nan")], [1.0, 1.0]),  # not finite
    ]:
        with pytest.raises(ValueError):
            p.set_spectrum_table(q, P)
    assert not p.has_spectrum_table()
    p.set_spectrum_table([0.1, 0.2, 0.3], [1.0, 2.0, 0.5])
    assert p.has_spectrum_table() and p.spectrum_q == [0.1, 0.2, 0.3]
    p.clear_spectrum_table()
    assert not p.has_spectrum_table()


def test_dense_table_reproduces_the_two_gaussians():
    a = _slices(_masker(_ice()))
    b = _slices(_masker(_ice(dense_table())))
    assert a.shape == b.shape and a.shape[0] > 10
    da, db = a - a.mean(axis=(1, 2), keepdims=True), b - b.mean(axis=(1, 2), keepdims=True)
    rel = float(np.max(np.abs(da - db)) / np.max(np.abs(da)))
    var_ratio = float(db.var() / da.var())
    print(f"\ndense table vs two Gaussians: max rel {rel:.2e}, variance ratio {var_ratio:.8f}")
    assert rel < 1e-4
    assert abs(var_ratio - 1) < 1e-5


def test_scale_of_P_does_not_matter():
    a = _slices(_masker(_ice(dense_table(1.0))))
    b = _slices(_masker(_ice(dense_table(4.0))))
    assert np.array_equal(a, b)


def test_default_is_unchanged_by_clearing():
    p = _ice(dense_table())
    p.clear_spectrum_table()
    assert np.array_equal(_slices(_masker(p)), _slices(_masker(_ice())))


def test_table_needs_the_band_limited_potential():
    os.environ["MULTEM_BANDLIMITED_POTENTIAL"] = "0"
    with pytest.raises(RuntimeError):
        _slices(_masker(_ice(dense_table())))


def test_single_region_honours_v0_and_m0():
    # the two hydrogen anchors sit in the slices unscaled, so the ratios fall slightly short
    a = _slices(_masker(_ice(v0=-1.0)))
    b = _slices(_masker(_ice(v0=4 * 10195.82)))
    da, db = a - a.mean(axis=(1, 2), keepdims=True), b - b.mean(axis=(1, 2), keepdims=True)
    ratio = float(db.std() / da.std())
    print(f"\nv0 x4: s.d. ratio {ratio:.6f} (2 expected, less the anchors)")
    assert abs(ratio - 2) < 3e-3
    p, q = _ice(), _ice()
    p.m0, q.m0 = 10.0, 20.0
    c, d = _slices(_masker(p)), _slices(_masker(q))
    mratio = float(d.mean() / c.mean())
    print(f"m0 10 -> 20: mean ratio {mratio:.6f} (2 expected, less the anchors)")
    assert abs(mratio - 2) < 3e-3
    assert abs(float(c.mean()) - float(a.mean())) > 1e-3 * abs(float(a.mean()))  # 10 is not the default


def test_multi_region_table():
    def masker(ice):
        m = multem.MultiRegionMasker(N, N, DR)
        shape = multem.Masker(N, N, DR)
        shape.set_cuboid((0.0, 0.0, 0.0), (N * DR, N * DR, LZ))
        m.add_region(shape, ice)
        m.set_seed(9)
        return m

    a = _psi(masker(_ice()))
    b = _psi(masker(_ice(dense_table())))
    rel = float(np.max(np.abs(a - b)) / np.max(np.abs(a - 1)))
    print(f"\nmulti-region dense table vs two Gaussians: max rel {rel:.2e}")
    assert rel < 1e-3
    c = _psi(masker(_ice(([0.05, 0.3, 0.6], [1.0, 2.0, 0.0]))))
    assert float(np.max(np.abs(c - a))) > 10 * float(np.max(np.abs(a - b)))


@pytest.mark.skipif(not multem.is_gpu_available(), reason="no GPU")
def test_table_on_the_gpu():
    a = _psi(_masker(_ice(dense_table())), "device")
    b = _psi(_masker(_ice(dense_table())), "device")
    g = _psi(_masker(_ice()), "device")
    h = _psi(_masker(_ice(dense_table())), "host")
    same = float(np.max(np.abs(a - b)))
    vs_gauss = float(np.max(np.abs(a - g)) / np.max(np.abs(g - 1)))
    vs_host = float(np.max(np.abs(a - h)) / np.max(np.abs(h - 1)))
    print(f"\nGPU table: repeat {same:.2e}, vs two Gaussians {vs_gauss:.2e}, vs host {vs_host:.2e}")
    assert same == 0
    assert vs_gauss < 1e-3
    assert vs_host < 1e-3
