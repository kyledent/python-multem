#
# test_keyed_grf.py
#
# The multi-region masker's opt-in keyed random field, slice origin and empty-region skip. With a slice
# origin and the keyed field, a slice's potential depends only on the regions present on it, their keys and
# the slice's place on the lattice: not on the order the regions were added in, not on regions beyond the
# grid, and not on the extent of the whole scene (the slicing atoms add nothing).
#
import os

import numpy as np
import pytest

import multem

N = 96
DR = 1.5
DZ = 3.0
Z_REF = 0.5
L = N * DR


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
    inp.spec_lx = L
    inp.spec_ly = L
    inp.spec_lz = 2000.0
    inp.spec_dz = DZ
    inp.iw_x = [L / 2]
    inp.iw_y = [L / 2]
    inp.spec_atoms = multem.AtomList([])
    return inp


def _ice(density):
    p = multem.IceParameters()
    p.density = density
    return p


def _shapes(far=()):
    """(name, masker, density, key): a slab holding two spheres it subtracts, plus spheres beyond the grid's
    columns at the given z"""
    slab = multem.Masker(N, N, DR)
    slab.set_cuboid((-50.0, -50.0, 10.0), (L + 100.0, L + 100.0, 60.0))
    out = [("slab", slab, 0.94, 101)]
    for name, (x, y, r), d, key in (("s1", (0.35, 0.4, 20.0), 1.0113, 102), ("s2", (0.65, 0.6, 16.0), 1.0818, 103)):
        s = multem.Masker(N, N, DR)
        s.set_sphere([x * L, y * L, 40.0], r)
        out.append((name, s, d, key))
    for j, z in enumerate(far):
        f = multem.Masker(N, N, DR)
        f.set_sphere([L + 2000.0 + 200.0 * j, L / 2, z], 30.0)
        out.append(("far%d" % j, f, 1.0, 200 + j))
    return out


def _masker(shapes, keyed=True, origin=True, skip=True):
    pos = {s[0]: k for k, s in enumerate(shapes)}
    m = multem.MultiRegionMasker(N, N, DR)
    for name, shape, density, key in shapes:
        sub = [pos["s1"], pos["s2"]] if name == "slab" else []
        m.add_region(shape, _ice(density), sub, key)
    m.set_seed(7)
    m.set_partial_volume(4)
    if origin:
        m.set_slice_origin(Z_REF, DZ)
    if keyed:
        m.set_keyed_grf(True)
        m.set_skip_empty(skip)
    return m


def _slices(masker, device="host"):
    sc = multem.SystemConfiguration()
    sc.precision = "float"
    sc.device = device
    out = {}

    def cb(z0, ze, V):
        k = int(np.floor((0.5 * (z0 + ze) - Z_REF) / DZ))
        out[k] = np.ascontiguousarray(np.array(V, dtype=np.float64)).tobytes()  # bytes: -0.0 != +0.0

    multem.compute_projected_potential(sc, _input(), masker, cb)
    return out


def _same_on_common(a, b):
    assert set(a) <= set(b)
    differ = [k for k in a if a[k] != b[k]]
    assert not differ, "slices differ at k = %s" % differ


def test_the_slice_origin_puts_the_extent_on_the_lattice():
    m = _masker(_shapes(), keyed=False)
    for z, raw, outward in ((m.zmin(), m.zmin_unsnapped(), -1), (m.zmax(), m.zmax_unsnapped(), 1)):
        k = (z - Z_REF) / DZ
        assert abs(k - round(k)) < 1e-9
        assert outward * (z - raw) >= 0 and abs(z - raw) < DZ
    m.clear_slice_origin()
    assert m.zmin() == m.zmin_unsnapped() and m.zmax() == m.zmax_unsnapped()


def test_keyed_slices_do_not_depend_on_the_order_of_the_regions():
    shapes = _shapes()
    _same_on_common(_slices(_masker(shapes)), _slices(_masker(shapes[::-1])))


def test_keyed_slices_do_not_depend_on_regions_beyond_the_grid():
    base = _slices(_masker(_shapes()))
    _same_on_common(base, _slices(_masker(_shapes(far=(400.0,)))))
    _same_on_common(base, _slices(_masker(_shapes(far=(-300.0,)))))


def test_skipping_an_empty_region_changes_nothing():
    shapes = _shapes(far=(400.0, -300.0))
    _same_on_common(_slices(_masker(shapes, skip=True)), _slices(_masker(shapes, skip=False)))


def test_the_slicing_atoms_add_nothing_with_a_slice_origin():
    # The scene's first and last slices hold the two atoms that mark its z range for slicing; with the regions
    # beyond the grid above and below, the same slices are interior and hold no atom
    plain = _slices(_masker(_shapes()))
    wide = _slices(_masker(_shapes(far=(400.0, -300.0))))
    for k in (min(plain), max(plain)):
        assert plain[k] == wide[k]


def test_the_keyed_field_refuses_two_regions_with_one_key():
    shapes = _shapes()
    name, shape, density, _ = shapes[2]
    shapes[2] = (name, shape, density, shapes[1][3])
    with pytest.raises(Exception, match="keys_unique"):
        _slices(_masker(shapes))


def test_gpu_agrees_with_host_for_the_keyed_field():
    host = _slices(_masker(_shapes()))
    try:
        dev = _slices(_masker(_shapes()), "device")
    except Exception as e:  # no device in this environment
        pytest.skip("no GPU: %s" % e)
    assert set(host) == set(dev)
    a = np.array([np.frombuffer(host[k], dtype=np.float64) for k in sorted(host)])
    b = np.array([np.frombuffer(dev[k], dtype=np.float64) for k in sorted(dev)])
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4 * np.abs(a).max())
