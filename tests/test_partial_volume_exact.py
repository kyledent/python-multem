#
# test_partial_volume_exact.py
#
# Spheres and cuboids (at any rotation) get their partial-volume fraction exactly, from the interval in
# which the ray along z meets the shape, in one evaluation: the sample count n does not change them.
# Cylinders keep the n sub-slice tests.
#
import numpy as np
import pytest

import multem

N = 48
DR = 1.5


def devices():
    return ["host"] + (["device"] if multem.is_gpu_available() else [])


def frac(m, zs, ze, n, device):
    return np.array(multem.compute_fraction_engine(m, zs, ze, n, device == "device")).reshape(N, N)


def sphere_reference(c, r, zs, ze):
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    h2 = r * r - ((i * DR - c[0]) ** 2 + (j * DR - c[1]) ** 2)
    h = np.sqrt(np.clip(h2, 0, None))
    lo = np.maximum(c[2] - h, zs)
    hi = np.minimum(c[2] + h, ze)
    return np.where((h2 > 0) & (hi > lo), (hi - lo) / (ze - zs), 0.0)


@pytest.mark.parametrize("device", devices())
def test_sphere_fraction_is_exact(device):
    c = [N * DR / 2 + 0.37, N * DR / 2 - 0.61, 31.3]
    r = 22.7
    m = multem.Masker(N, N, DR)
    m.set_sphere(c, r)
    for zs in np.arange(0.0, 60.0, 2.9):
        ref = sphere_reference(c, r, zs, zs + 3.0)
        got = frac(m, zs, zs + 3.0, 4, device)
        assert np.max(np.abs(got - ref)) < 1e-12, zs


@pytest.mark.parametrize("device", devices())
def test_exact_shapes_do_not_depend_on_n(device):
    s = multem.Masker(N, N, DR)
    s.set_sphere([N * DR / 2, N * DR / 2, 20.0], 15.0)
    b = multem.Masker(N, N, DR)
    b.set_cuboid((10.0, 12.0, 8.3), (40.0, 35.0, 21.7))
    b.set_rotation([N * DR / 2, N * DR / 2, 20.0], [0.0, np.deg2rad(37.0), 0.0])
    for m in (s, b):
        for zs in (5.0, 9.1, 17.4, 26.0):
            assert np.array_equal(frac(m, zs, zs + 3.0, 1, device), frac(m, zs, zs + 3.0, 64, device))


@pytest.mark.parametrize("device", devices())
def test_rotated_cuboid_fraction_matches_fine_sampling(device):
    # the exact interval against the point test sampled very finely (the cylinder path's method)
    b = multem.Masker(N, N, DR)
    b.set_cuboid((10.0, 12.0, 8.3), (40.0, 35.0, 21.7))
    b.set_rotation([N * DR / 2, N * DR / 2, 20.0], [np.deg2rad(11.0), np.deg2rad(37.0), 0.0])
    n_fine = 2048
    total_exact = total_fine = 0.0
    for zs in np.arange(-10.0, 50.0, 3.0):
        exact = frac(b, zs, zs + 3.0, 4, device)
        fine = np.zeros((N, N))
        h = 3.0 / n_fine
        for k in range(n_fine):
            fine += np.array(b.compute(zs + k * h, zs + (k + 1) * h)).reshape(N, N)
        fine /= n_fine
        assert np.max(np.abs(exact - fine)) <= 1.0 / n_fine + 1e-12, zs
        total_exact += exact.sum()
        total_fine += fine.sum()
    assert total_exact > 100


@pytest.mark.parametrize("device", devices())
def test_cuboid_path_is_the_box_height_through_the_slab(device):
    # an axis-aligned box: every pixel inside its footprint has path = the box height over the whole z range
    b = multem.Masker(N, N, DR)
    b.set_cuboid((10.0, 12.0, 8.3), (40.0, 35.0, 21.7))
    path = sum(frac(b, zs, zs + 3.0, 2, device) * 3.0 for zs in np.arange(0.0, 36.0, 3.0))
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    inside = (i * DR >= 10.0) & (i * DR < 50.0) & (j * DR >= 12.0) & (j * DR < 47.0)
    assert np.max(np.abs(path[inside] - 21.7)) < 1e-12
    assert np.all(path[~inside] == 0)


def test_device_equals_host_bit_for_bit():
    if not multem.is_gpu_available():
        pytest.skip("no GPU")
    s = multem.Masker(N, N, DR)
    s.set_sphere([N * DR / 2 + 0.3, N * DR / 2 - 0.2, 20.0], 25.0)
    b = multem.Masker(N, N, DR)
    b.set_cuboid((10.0, 12.0, 8.3), (40.0, 35.0, 21.7))
    b.set_rotation([N * DR / 2, N * DR / 2, 20.0], [np.deg2rad(11.0), np.deg2rad(37.0), 0.0])
    for m in (s, b):
        for zs in np.arange(-10, 50, 2.7):
            assert np.array_equal(frac(m, zs, zs + 3.0, 16, "host"), frac(m, zs, zs + 3.0, 16, "device")), zs


def test_cylinder_still_samples():
    c = multem.Masker(N, N, DR)
    # axis along x at z = 20, radius 15: the slice [33, 36) holds the curved top surface (z = 35 on the axis),
    # so the sampled fraction takes values between 0 and 1 there, and n = 1 is the 0/1 midpoint test
    c.set_cylinder([0.0, N * DR / 2, 20.0], [1, 0, 0], N * DR, [15.0], [0], [0])
    f1 = frac(c, 33.0, 36.0, 1, "host")
    f8 = frac(c, 33.0, 36.0, 8, "host")
    assert set(np.unique(f1)) <= {0.0, 1.0}
    assert f1.sum() > 0
    assert np.any((f8 > 0) & (f8 < 1))
    assert set(np.unique(np.round(f8 * 8, 9))) <= set(range(9))
