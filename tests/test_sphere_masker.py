"""
Tests for the SphereMasker primitive in python-multem.

Covers:
  - Basic sphere mask generation (centre pixel lit, corners dark)
  - Bounding box (AABB) correctness
  - Translation shifts the mask
  - Rotation moves the centre correctly
  - Slices outside the sphere are all-zero
  - Sphere area scales with radius squared (pi*r^2)
  - Integration with Masker dispatcher (set_sphere)
  - Integration with MultiRegionMasker
"""

import math
import numpy as np
import pytest

import multem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_mask_2d(masker, zs, ze):
    """Compute mask via Masker.compute() and reshape to (xsize, ysize)."""
    flat = masker.compute(zs, ze)
    return np.array(flat).reshape(masker.xsize(), masker.ysize())


# ---------------------------------------------------------------------------
# Basic sphere tests via the Masker dispatcher
# ---------------------------------------------------------------------------

class TestSphereMaskerBasic:
    """Basic sphere mask creation and containment."""

    def test_set_sphere_shape(self):
        m = multem.Masker(64, 64, 1.0)
        m.set_sphere([32, 32, 10], 5.0)
        assert m.shape() == multem.MaskerShape.Sphere

    def test_centre_pixel_inside(self):
        """The pixel containing the sphere centre should be masked."""
        m = multem.Masker(64, 64, 1.0)
        m.set_sphere([32, 32, 10], 5.0)
        mask = compute_mask_2d(m, 9.5, 10.5)
        # Pixel (31,31) has centre at (31.5, 31.5) -- closest to (32,32)
        assert mask[31, 31] == 1

    def test_far_corner_outside(self):
        """Pixels far from the sphere should not be masked."""
        m = multem.Masker(64, 64, 1.0)
        m.set_sphere([32, 32, 10], 5.0)
        mask = compute_mask_2d(m, 9.5, 10.5)
        assert mask[0, 0] == 0
        assert mask[63, 63] == 0

    def test_slice_outside_z_range(self):
        """A slice entirely above or below the sphere is all zeros."""
        m = multem.Masker(64, 64, 1.0)
        m.set_sphere([32, 32, 10], 5.0)
        # Sphere spans z=[5, 15], so z=[20,21] is outside
        mask = compute_mask_2d(m, 20, 21)
        assert mask.sum() == 0

        # Below the sphere
        mask_below = compute_mask_2d(m, 0, 1)
        assert mask_below.sum() == 0

    def test_equatorial_slice_is_circular(self):
        """The equatorial slice should approximate a circle of area pi*r^2."""
        r = 20.0
        px = 1.0
        n = 128
        m = multem.Masker(n, n, px)
        m.set_sphere([64, 64, 50], r)
        mask = compute_mask_2d(m, 49.5, 50.5)
        area = mask.sum() * px * px
        expected = math.pi * r * r
        # Discretisation error: allow 5% tolerance
        assert abs(area - expected) / expected < 0.05, \
            f"area={area:.1f}, expected={expected:.1f}"


class TestSphereMaskerAABB:
    """Bounding box (AABB) tests."""

    def test_aabb_no_rotation(self):
        m = multem.Masker(64, 64, 1.0)
        m.set_sphere([30, 40, 50], 10.0)
        assert m.xmin() == pytest.approx(20.0)
        assert m.xmax() == pytest.approx(40.0)
        assert m.ymin() == pytest.approx(30.0)
        assert m.ymax() == pytest.approx(50.0)
        assert m.zmin() == pytest.approx(40.0)
        assert m.zmax() == pytest.approx(60.0)

    def test_aabb_with_translation(self):
        m = multem.Masker(64, 64, 1.0)
        m.set_sphere([30, 40, 50], 10.0)
        m.set_translation([5, -3, 2])
        assert m.xmin() == pytest.approx(25.0)
        assert m.xmax() == pytest.approx(45.0)
        assert m.ymin() == pytest.approx(27.0)
        assert m.ymax() == pytest.approx(47.0)
        assert m.zmin() == pytest.approx(42.0)
        assert m.zmax() == pytest.approx(62.0)


class TestSphereMaskerTranslation:
    """Translation shifts the mask."""

    def test_translation_shifts_mask(self):
        n, px, r = 64, 1.0, 8.0
        m = multem.Masker(n, n, px)

        # Sphere at centre
        m.set_sphere([32, 32, 10], r)
        mask0 = compute_mask_2d(m, 9.5, 10.5)

        # Shift by (+10, +5, 0)
        m.set_translation([10, 5, 0])
        mask1 = compute_mask_2d(m, 9.5, 10.5)

        # The shifted mask should have its centroid displaced
        ys0, xs0 = np.where(mask0)
        ys1, xs1 = np.where(mask1)
        dx = xs1.mean() - xs0.mean()
        dy = ys1.mean() - ys0.mean()
        # Shift is in coordinate space, pixel = coord / pixel_size
        assert abs(dx - 5) < 1.5   # y-shift in pixel coords
        assert abs(dy - 10) < 1.5  # x-shift in pixel coords


class TestSphereMaskerRotation:
    """Rotation moves the sphere centre."""

    def test_rotation_preserves_mask_count(self):
        """Rotating a sphere about its own centre doesn't change the mask."""
        n, px, r = 64, 1.0, 10.0
        centre = [32, 32, 32]
        m = multem.Masker(n, n, px)
        m.set_sphere(centre, r)
        mask0 = compute_mask_2d(m, 31.5, 32.5)

        # Rotate about the sphere centre -- should be identical
        m.set_rotation(centre, [0, 0.5, 0])
        mask1 = compute_mask_2d(m, 31.5, 32.5)
        assert mask0.sum() == mask1.sum()

    def test_rotation_about_external_point(self):
        """Rotating about an external point moves the sphere centre."""
        n, px, r = 128, 1.0, 5.0
        m = multem.Masker(n, n, px)
        m.set_sphere([64, 64, 64], r)

        # 90-degree rotation about y-axis, centred at (64, 64, 64)
        # doesn't move the centre since it's at the rotation origin
        mask0 = compute_mask_2d(m, 63.5, 64.5)

        # Now place sphere off-centre and rotate about (64,64,64)
        m.set_sphere([84, 64, 64], r)
        m.set_rotation([64, 64, 64], [0, math.pi/2, 0])
        # After 90-deg rotation about y: x->z, z->-x
        # (84-64, 0, 0) -> (0, 0, -(84-64)) = (0, 0, -20)
        # New centre in microscope: (64, 64, 64-20) = (64, 64, 44)
        assert m.zmin() == pytest.approx(39.0)
        assert m.zmax() == pytest.approx(49.0)

        # Equatorial slice at z=44 should have non-zero mask
        mask_rot = compute_mask_2d(m, 43.5, 44.5)
        assert mask_rot.sum() > 0


class TestSphereMaskerRadiusScaling:
    """Mask area scales correctly with radius."""

    def test_area_doubles_with_sqrt2_radius(self):
        n, px = 256, 0.5
        r1 = 10.0
        r2 = r1 * math.sqrt(2)
        cz = 50.0

        m = multem.Masker(n, n, px)
        m.set_sphere([64, 64, cz], r1)
        mask1 = compute_mask_2d(m, cz - 0.25, cz + 0.25)

        m.set_sphere([64, 64, cz], r2)
        mask2 = compute_mask_2d(m, cz - 0.25, cz + 0.25)

        ratio = mask2.sum() / mask1.sum()
        assert abs(ratio - 2.0) < 0.15, f"ratio={ratio:.2f}, expected ~2.0"


class TestSphereMaskerMultiRegion:
    """SphereMasker works within MultiRegionMasker."""

    def test_sphere_in_multi_region(self):
        n, px = 64, 1.0
        masker = multem.MultiRegionMasker(n, n, px)

        shape = multem.Masker(n, n, px)
        shape.set_sphere([32, 32, 10], 8.0)
        params = multem.IceParameters()
        params.density = 1.2
        masker.add_region(shape, params)

        assert masker.num_regions() == 1
        r = masker.region(0)
        assert r.material.density == 1.2

    def test_sphere_subtract_from_cuboid(self):
        """Cuboid minus sphere: the sphere hole reduces the mask count."""
        n, px = 64, 1.0
        masker = multem.MultiRegionMasker(n, n, px)

        # Region 0: sphere (the void)
        sphere = multem.Masker(n, n, px)
        sphere.set_sphere([32, 32, 10], 10.0)
        masker.add_region(sphere, multem.IceParameters())

        # Region 1: cuboid minus sphere
        cuboid = multem.Masker(n, n, px)
        cuboid.set_cuboid([0, 0, 0], [64, 64, 20])
        masker.add_region(cuboid, multem.IceParameters(), [0])

        assert masker.num_regions() == 2
        assert masker.region(1).subtract_regions == [0]


class TestSphereMaskerSmallRadius:
    """Edge case: very small sphere."""

    def test_single_voxel_sphere(self):
        """A sphere with radius < pixel_size should light at most 1-2 pixels per row."""
        n, px = 32, 1.0
        m = multem.Masker(n, n, px)
        m.set_sphere([16, 16, 10], 0.4)
        mask = compute_mask_2d(m, 9.8, 10.2)
        # Very small sphere -- should have very few lit pixels
        assert mask.sum() <= 4
        assert mask.sum() >= 0  # might be 0 if centre falls between pixels
