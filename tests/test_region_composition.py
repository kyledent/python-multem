#
# test_region_composition.py
#
# How a region's partial-volume fraction takes off the regions it subtracts. The interval composition (the
# default) gives each pixel the region's own part of the column through the slice, less the union of the
# subtracted regions' parts; the legacy composition subtracts each one's whole fraction. They agree where
# every subtracted part lies inside the region's own part, and differ where a subtracted region only abuts
# the region, or where two subtracted regions nest or overlap: there the legacy composition takes material
# the region owns.
#
import numpy as np
import pytest

import multem

N = 64
DR = 1.0
L = N * DR
C = 32.0  # a pixel corner: the column at pixel (32, 32) is x = y = 32


def devices():
    return ["host"] + (["device"] if multem.is_gpu_available() else [])


def ice():
    p = multem.IceParameters()
    p.density = 0.94
    return p


def layer(z0, z1):
    """A cuboid covering every pixel between z0 and z1"""
    m = multem.Masker(N, N, DR)
    m.set_cuboid((-50.0, -50.0, z0), (L + 100.0, L + 100.0, z1 - z0))
    return m


def ellipsoid_A(semi):
    return np.diag(1.0 / np.asarray(semi, float) ** 2).ravel().tolist()


def ball(r, centre=(C, C, 20.0)):
    m = multem.Masker(N, N, DR)
    m.set_convex(ellipsoid_A([r, r, r]), list(centre), [])
    return m


def shell(r_out, r_in, centre=(C, C, 20.0)):
    m = multem.Masker(N, N, DR)
    m.set_convex(ellipsoid_A([r_out] * 3), list(centre), [], True, ellipsoid_A([r_in] * 3), list(centre), [])
    return m


def fraction(shapes, subtract, zs, ze, device, mode="interval", region=0):
    """Region `region`'s composed fraction of [zs, ze), per pixel (N, N)"""
    m = multem.MultiRegionMasker(N, N, DR)
    for k, s in enumerate(shapes):
        m.add_region(s, ice(), subtract.get(k, []))
    m.set_partial_volume(4)
    m.set_region_composition(mode)
    return np.array(multem.compute_region_fraction_engine(m, region, zs, ze, device == "device")).reshape(N, N)


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize(
    "case, shapes, zs, ze, interval, legacy",
    [
        # self [10, 20), sub [20, 30) abuts it; the slice [18, 22) holds 2 A of each
        ("abut", lambda: [layer(10, 20), layer(20, 30)], 18.0, 22.0, 0.5, 0.0),
        # sub [15, 18) inside self [10, 30), slice [14, 20): 3 of 6 A left either way
        ("nest", lambda: [layer(10, 30), layer(15, 18)], 14.0, 20.0, 0.5, 0.5),
        # subs [15, 19) and [16, 17) inside it: the inner one is inside the outer one
        ("double nest", lambda: [layer(10, 30), layer(15, 19), layer(16, 17)], 14.0, 20.0, 2 / 6, 1 / 6),
        # subs [15, 18) and [17, 19) overlap each other: their union is [15, 19)
        ("overlap", lambda: [layer(10, 30), layer(15, 18), layer(17, 19)], 14.0, 20.0, 2 / 6, 1 / 6),
    ],
)
def test_the_composition_of_layers(device, case, shapes, zs, ze, interval, legacy):
    subtract = {0: list(range(1, len(shapes())))}
    got = fraction(shapes(), subtract, zs, ze, device)
    old = fraction(shapes(), subtract, zs, ze, device, mode="legacy")
    np.testing.assert_allclose(got, interval, rtol=0, atol=1e-12, err_msg=case)
    np.testing.assert_allclose(old, legacy, rtol=0, atol=1e-12, err_msg=case)


@pytest.mark.parametrize("device", devices())
def test_a_shell_subtracted_from_a_layer_and_from_its_own_hole(device):
    # On the column through the centre the shell (r 10, hole r 6) is [10, 14) and [26, 30); the slice [8, 32)
    # is 24 A. A layer holding the shell keeps 16 of 24 A either way; the ball filling the hole, [14, 26),
    # only abuts the shell and keeps all of its 12 A, where the legacy composition leaves it 4
    i = int(C / DR)
    holder = fraction([layer(0, 40), shell(10, 6)], {0: [1]}, 8.0, 32.0, device)
    hole = fraction([ball(6), shell(10, 6)], {0: [1]}, 8.0, 32.0, device)
    hole_old = fraction([ball(6), shell(10, 6)], {0: [1]}, 8.0, 32.0, device, mode="legacy")
    assert abs(holder[i, i] - 16 / 24) < 1e-12
    assert abs(hole[i, i] - 12 / 24) < 1e-12
    assert abs(hole_old[i, i] - 4 / 24) < 1e-12


@pytest.mark.parametrize("device", devices())
def test_a_region_no_subtracted_part_meets_is_composed_bit_for_bit_as_before(device):
    # A ball in a slab, with a second ball far below the slice it subtracts: the subtracted ball reaches the
    # slice's z range nowhere near the first, so every pixel takes the region's own fraction, as legacy
    shapes = lambda: [ball(9.3, (C + 0.3, C - 0.2, 20.0)), ball(5.0, (10.0, 10.0, 17.0))]  # noqa: E731
    new = fraction(shapes(), {0: [1]}, 16.5, 19.5, device)
    old = fraction(shapes(), {0: [1]}, 16.5, 19.5, device, mode="legacy")
    far = np.hypot(*np.meshgrid(np.arange(N) * DR - 10.0, np.arange(N) * DR - 10.0, indexing="ij")) > 5.0
    assert np.array_equal(new[far].view(np.int64), old[far].view(np.int64))
    assert (new[far] > 0).sum() > 100


@pytest.mark.parametrize("device", devices())
def test_a_partition_sums_to_one_at_every_pixel(device):
    # A slab holding two cut balls that abut along a plane, one at a higher rank, each with a nested child,
    # and a surface blob outranked by everything (as subtract lists from a ranked scene give them): through
    # every slice, the regions' fractions sum to the fraction of the column the union covers
    def cut_ball(r, centre, normal, offset):
        m = multem.Masker(N, N, DR)
        m.set_convex(ellipsoid_A([r, r, r]), list(centre), [(*normal, offset)])
        return m

    tilt = (0.0, np.radians(35.0), 0.0)
    shapes = [
        layer(5.0, 45.0),                                               # 0 slab (lowest rank)
        cut_ball(20.0, (C - 6.0, C, 25.0), (1.0, 0.0, 0.0), C + 0.37),  # 1 x <= C + 0.37
        cut_ball(20.0, (C + 6.0, C, 25.0), (-1.0, 0.0, 0.0), -(C + 0.37)),  # 2 x >= C + 0.37, abuts 1
        ball(4.0, (C - 10.0, C + 2.0, 24.0)),                           # 3 child of 1
        ball(3.0, (C + 9.0, C - 3.0, 27.0)),                            # 4 child of 2
        ball(9.0, (C, C + 5.0, 46.0)),                                  # 5 surface blob, lowest of all
    ]
    # ranks: 5 < 0 < 1 < 2 < 3, 4; each subtracts every higher one (the scene's paint plan)
    rank = {5: 0, 0: 1, 1: 2, 2: 3, 3: 4, 4: 4}
    subtract = {k: [o for o in rank if rank[o] > rank[k]] for k in rank}
    centre = [C, C, 25.0]
    for s in shapes:
        s.set_rotation(centre, tilt)
    m = multem.MultiRegionMasker(N, N, DR)
    for k, s in enumerate(shapes):
        m.add_region(s, ice(), subtract[k])
    m.set_partial_volume(4)
    worst = {"interval": 0.0, "legacy": 0.0}
    for mode in worst:
        m.set_region_composition(mode)
        for z in np.arange(4.0, 56.0, 3.0):
            total = sum(np.array(multem.compute_region_fraction_engine(m, k, z, z + 3.0, device == "device"))
                        for k in range(len(shapes)))
            # the union's own fraction: the slab or the blob or a ball (the rest lie inside the slab)
            cover = np.zeros(N * N)
            for zz in np.linspace(z, z + 3.0, 3001)[:-1] + 0.0005:
                inside = np.array(multem.compute_mask_engine(shapes[0], zz - 1e-4, zz + 1e-4, False)) \
                    | np.array(multem.compute_mask_engine(shapes[5], zz - 1e-4, zz + 1e-4, False))
                cover += inside / 3000.0
            worst[mode] = max(worst[mode], float(np.abs(total - cover).max()))
    # the reference is a 3000-point midpoint sum: 1/3000 per face
    assert worst["interval"] < 2e-3, worst
    assert worst["legacy"] > 0.1, worst
