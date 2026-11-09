#
# test_convex_region.py
#
# Masker.set_convex: an optional quadric (p - c)^T A (p - c) <= 1 intersected with half-spaces n.p <= d, minus an
# optional inner set. Ellipsoids (axes, cuts), boxes at any rotation, slabs (rectangle or disc footprint) and
# ellipsoidal shells are all of this form. The ray along z meets each set in one interval, so masks and exact
# partial-volume fractions come from the same code. Also the test hook compute_region_labels_engine.
#
import numpy as np
import pytest

import multem

N = 40
DR = 1.5
L = N * DR


def devices():
    return ["host"] + (["device"] if multem.is_gpu_available() else [])


def frac(m, zs, ze, device="host", n=4):
    return np.array(multem.compute_fraction_engine(m, zs, ze, n, device == "device")).reshape(N, N)


def mask(m, zs, ze, device="host"):
    return np.array(multem.compute_mask_engine(m, zs, ze, device == "device")).reshape(N, N)


def rot(axis, deg):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.deg2rad(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def ellipsoid_A(semi, U=np.eye(3)):
    return (U @ np.diag(1.0 / np.asarray(semi, float) ** 2) @ U.T).ravel().tolist()


def box_planes(c, half, U=np.eye(3)):
    c = np.asarray(c, float)
    out = []
    for k in range(3):
        a = U[:, k]
        out.append((*a, half[k] + a @ c))
        out.append((*(-a), half[k] - a @ c))
    return out


def ref_interval(A, c, planes, x, y):
    """An independent reference: the interval along z by bisection on the inequalities, to ~1e-12."""
    A = None if A is None else np.asarray(A).reshape(3, 3)
    def inside(z):
        p = np.array([x, y, z])
        if A is not None and (p - c) @ A @ (p - c) > 1:
            return False
        return all(np.dot(pl[:3], p) <= pl[3] for pl in planes)
    zs = np.linspace(-200, 200, 4001)
    ins = [z for z in zs if inside(z)]
    if not ins:
        return None
    def edge(a, b):
        for _ in range(80):
            m = 0.5 * (a + b)
            if inside(m) == inside(a):
                a = m
            else:
                b = m
        return 0.5 * (a + b)
    return edge(ins[0] - 0.1, ins[0]), edge(ins[-1], ins[-1] + 0.1)


@pytest.mark.parametrize("device", devices())
def test_ellipsoid_as_sphere_matches_the_sphere(device):
    c = [L / 2 + 0.3, L / 2 - 0.4, 25.0]
    r = 17.3
    s = multem.Masker(N, N, DR)
    s.set_sphere(c, r)
    e = multem.Masker(N, N, DR)
    e.set_convex(ellipsoid_A([r, r, r]), c, [])
    assert e.shape() == multem.MaskerShape.Convex
    assert abs(e.zmin() - s.zmin()) < 0.5 and abs(e.zmax() - s.zmax()) < 0.5
    for zs in np.arange(0.0, 50.0, 2.9):
        assert np.max(np.abs(frac(e, zs, zs + 3.0, device) - frac(s, zs, zs + 3.0, device))) < 1e-12


@pytest.mark.parametrize("device", devices())
def test_rotated_box_matches_the_rotated_cuboid(device):
    offset, size = np.array([12.0, 14.0, 9.0]), np.array([30.0, 26.0, 18.0])
    centre = [L / 2, L / 2, 20.0]
    rv = np.deg2rad(31.0) * np.array([0.3, 1.0, 0.2]) / np.linalg.norm([0.3, 1.0, 0.2])
    b = multem.Masker(N, N, DR)
    b.set_cuboid(tuple(offset), tuple(size))
    b.set_rotation(centre, rv.tolist())
    v = multem.Masker(N, N, DR)
    v.set_convex([], [0, 0, 0], box_planes(offset + size / 2, size / 2))
    v.set_rotation(centre, rv.tolist())
    for zs in np.arange(-10.0, 50.0, 3.0):
        assert np.max(np.abs(frac(v, zs, zs + 3.0, device) - frac(b, zs, zs + 3.0, device))) < 1e-9, zs


def test_rotated_cut_ellipsoid_against_a_reference():
    U = rot([1, 2, 0.5], 40)
    c = np.array([L / 2, L / 2 + 2, 22.0])
    semi = [20.0, 12.0, 9.0]
    cut = U[:, 0]
    planes = [(*cut, 8.0 + cut @ c)]                  # keep n.(p - c) <= 8 along the long axis
    A = ellipsoid_A(semi, U)
    m = multem.Masker(N, N, DR)
    m.set_convex(A, c.tolist(), planes)
    paths = sum(frac(m, zs, zs + 2.0) * 2.0 for zs in np.arange(-20.0, 64.0, 2.0))
    rng = np.random.default_rng(3)
    for _ in range(40):
        i, j = rng.integers(0, N, 2)
        iv = ref_interval(A, c, planes, i * DR, j * DR)
        ref = 0.0 if iv is None else iv[1] - iv[0]
        assert abs(paths[i, j] - ref) < 1e-6, (i, j, paths[i, j], ref)


@pytest.mark.parametrize("device", devices())
def test_shell_is_outer_minus_inner(device):
    c = [L / 2, L / 2, 30.0]
    outer, inner = [22.0, 18.0, 15.0], [19.0, 15.0, 12.0]
    sh = multem.Masker(N, N, DR)
    sh.set_convex(ellipsoid_A(outer), c, [], True, ellipsoid_A(inner), c, [])
    path = sum(frac(sh, zs, zs + 3.0, device) * 3.0 for zs in np.arange(0.0, 60.0, 3.0))
    # through the centre column: twice the wall along z
    i0 = j0 = int(round(L / 2 / DR))
    assert abs(path[i0, j0] - 2 * (outer[2] - inner[2])) < 1e-9
    # the 0/1 mask at the centre slice is empty at the centre (inside the inner ellipsoid)
    assert mask(sh, 28.5, 31.5, device)[i0, j0] == 0


@pytest.mark.parametrize("device", devices())
def test_tilted_disc_slab_path(device):
    # a 20 A slab tilted 25 deg, disc footprint radius 18: path = t / cos(tilt) inside the disc's shadow
    n = rot([0, 1, 0], 25) @ np.array([0, 0, 1.0])
    o = np.array([L / 2, L / 2, 25.0])
    t = 20.0
    planes = [(*n, t / 2 + n @ o), (*(-n), t / 2 - n @ o)]
    A = ((np.eye(3) - np.outer(n, n)) / 18.0 ** 2).ravel().tolist()
    m = multem.Masker(N, N, DR)
    m.set_convex(A, o.tolist(), planes)
    path = sum(frac(m, zs, zs + 3.0, device) * 3.0 for zs in np.arange(-10.0, 60.0, 3.0))
    i0 = j0 = int(round(L / 2 / DR))
    assert abs(path[i0, j0] - t / n[2]) < 1e-9
    assert path[0, 0] == 0


def test_device_equals_host_bit_for_bit():
    if not multem.is_gpu_available():
        pytest.skip("no GPU")
    U = rot([1, 2, 0.5], 40)
    c = [L / 2, L / 2 + 2, 22.0]
    m = multem.Masker(N, N, DR)
    m.set_convex(ellipsoid_A([20.0, 12.0, 9.0], U), c, [(*U[:, 0], 8.0 + U[:, 0] @ np.array(c))], True,
                 ellipsoid_A([16.0, 9.0, 6.0], U), c, [])
    for zs in np.arange(0.0, 45.0, 2.3):
        assert np.array_equal(frac(m, zs, zs + 3.0, "host"), frac(m, zs, zs + 3.0, "device")), zs
        assert np.array_equal(mask(m, zs, zs + 3.0, "host"), mask(m, zs, zs + 3.0, "device")), zs
        assert np.array_equal(mask(m, zs, zs + 3.0, "host"), np.array(m.compute(zs, zs + 3.0)).reshape(N, N)), zs


def test_extent_and_nothing_in_view():
    m = multem.Masker(N, N, DR)
    m.set_convex(ellipsoid_A([5.0, 5.0, 7.0]), [L / 2, L / 2, 30.0], [])
    assert m.zmin() == pytest.approx(23.0, abs=1e-9) and m.zmax() == pytest.approx(37.0, abs=1e-9)
    far = multem.Masker(N, N, DR)
    far.set_convex(ellipsoid_A([5.0, 5.0, 7.0]), [L + 100, L / 2, 30.0], [])
    assert far.zmin() > far.zmax()
    assert mask(far, 28.0, 31.0).sum() == 0


def test_region_labels():
    ice = multem.IceParameters()
    ice.density = 0.94
    slab = multem.Masker(N, N, DR)
    slab.set_cuboid((0.0, 0.0, 10.0), (L, L, 30.0))
    ball = multem.Masker(N, N, DR)
    ball.set_convex(ellipsoid_A([10.0, 10.0, 10.0]), [L / 2, L / 2, 25.0], [])
    m = multem.MultiRegionMasker(N, N, DR)
    m.add_region(slab, ice, [1])
    m.add_region(ball, ice)
    lab = np.array(multem.compute_region_labels_engine(m, 23.5, 26.5)).reshape(N, N)
    i0 = int(round(L / 2 / DR))
    assert lab[i0, i0] == 1 and lab[0, 0] == 0 and not np.any(lab == -2)
    # without the subtraction, the ball overlaps the slab: -2 there
    m2 = multem.MultiRegionMasker(N, N, DR)
    m2.add_region(slab, ice)
    m2.add_region(ball, ice)
    lab2 = np.array(multem.compute_region_labels_engine(m2, 23.5, 26.5)).reshape(N, N)
    assert lab2[i0, i0] == -2 and lab2[0, 0] == 0
    if multem.is_gpu_available():
        assert np.array_equal(lab, np.array(multem.compute_region_labels_engine(m, 23.5, 26.5, True)).reshape(N, N))


@pytest.mark.parametrize("device", devices())
def test_convex_region_in_the_projected_potential(device):
    # a sphere region and the same sphere as a convex region give the same slices
    def run(as_convex):
        w = multem.IceParameters()
        w.density = 0.94
        w.v0 = 1e-9
        d = multem.IceParameters()
        d.density = 1.35
        d.v0 = 1e-9
        slab = multem.Masker(N, N, DR)
        slab.set_cuboid((0.0, 0.0, 10.0), (L, L, 40.0))
        ball = multem.Masker(N, N, DR)
        c, r = [L / 2 + 0.2, L / 2 - 0.3, 30.0], 12.0
        if as_convex:
            ball.set_convex(ellipsoid_A([r, r, r]), c, [])
        else:
            ball.set_sphere(c, r)
        m = multem.MultiRegionMasker(N, N, DR)
        m.add_region(slab, w, [1])
        m.add_region(ball, d)
        m.set_seed(1)
        m.set_partial_volume(4)
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
        inp.spec_lx = inp.spec_ly = L
        inp.spec_lz = 60.0
        inp.spec_dz = 3.0
        inp.iw_x = [L / 2]
        inp.iw_y = [L / 2]
        inp.spec_atoms = multem.AtomList([(1, 1.0, 1.0, 0.0, 0.0, 1.0, 0, 0), (1, 1.0, 1.0, 60.0, 0.0, 1.0, 0, 0)])
        sc = multem.SystemConfiguration()
        sc.precision = "double"
        sc.device = device
        V = []
        multem.compute_projected_potential(sc, inp, m, lambda a, b, v: V.append(np.array(v)))
        return np.sum(V, axis=0)
    a, b = run(False), run(True)
    assert np.max(np.abs(a - b)) < 1e-9 * np.max(np.abs(a))
