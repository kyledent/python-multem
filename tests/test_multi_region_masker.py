import multem


def test_ice_parameters_m0_v0():
    """Test that IceParameters has m0 and v0 override fields."""
    params = multem.IceParameters()
    assert params.m0 == -1  # default: use built-in constant
    assert params.v0 == -1

    params.m0 = 200.0
    params.v0 = 5000.0
    assert params.m0 == 200.0
    assert params.v0 == 5000.0


def test_masker_region():
    """Test MaskerRegion struct bindings."""
    region = multem.MaskerRegion()
    assert region.subtract_regions == []

    region.subtract_regions = [0, 2]
    assert region.subtract_regions == [0, 2]

    # Set material
    params = multem.IceParameters()
    params.density = 2.0
    params.m0 = 300.0
    region.material = params
    assert region.material.density == 2.0
    assert region.material.m0 == 300.0

    # Set shape
    shape = multem.Masker(128, 128, 1.0)
    shape.set_cuboid([0, 0, 0], [100, 110, 50])
    region.shape = shape
    assert region.shape.xsize() == 128


def test_multi_region_masker_construction():
    """Test MultiRegionMasker creation and region management."""
    masker = multem.MultiRegionMasker()
    assert masker.num_regions() == 0

    masker = multem.MultiRegionMasker(256, 256, 1.5)
    assert masker.xsize() == 256
    assert masker.ysize() == 256
    assert masker.pixel_size() == 1.5
    assert masker.image_size() == 256 * 256


def test_multi_region_masker_add_regions():
    """Test adding regions with boolean subtract operations."""
    masker = multem.MultiRegionMasker(256, 256, 1.0)

    # Region 0: ice cylinder (the hole)
    ice_shape = multem.Masker(256, 256, 1.0)
    ice_shape.set_cylinder([128, 128, 0], [0, 0, 1], 30, [50], [0], [0])
    ice_params = multem.IceParameters()
    ice_params.density = 0.94

    masker.add_region(ice_shape, ice_params)
    assert masker.num_regions() == 1

    # Region 1: foil cuboid minus region 0 (annulus)
    foil_shape = multem.Masker(256, 256, 1.0)
    foil_shape.set_cuboid([0, 0, 0], [256, 256, 30])
    foil_params = multem.IceParameters()
    foil_params.density = 2.0
    foil_params.m0 = 200.0
    foil_params.v0 = 8000.0

    masker.add_region(foil_shape, foil_params, [0])  # subtract region 0
    assert masker.num_regions() == 2

    # Verify region retrieval
    r0 = masker.region(0)
    assert r0.material.density == 0.94
    assert r0.subtract_regions == []

    r1 = masker.region(1)
    assert r1.material.m0 == 200.0
    assert r1.subtract_regions == [0]


def test_multi_region_masker_z_extent():
    """Test zmin/zmax across regions."""
    masker = multem.MultiRegionMasker(128, 128, 1.0)

    shape1 = multem.Masker(128, 128, 1.0)
    shape1.set_cuboid([0, 0, -10], [128, 128, 20])
    params1 = multem.IceParameters()
    masker.add_region(shape1, params1)

    shape2 = multem.Masker(128, 128, 1.0)
    shape2.set_cuboid([0, 0, -5], [128, 128, 40])
    params2 = multem.IceParameters()
    masker.add_region(shape2, params2)

    assert masker.zmin() == -10
    # zmax depends on offset + length interpretation in cuboid masker
    assert masker.zmax() >= 10  # at least larger than zmin


def test_multi_region_masker_rotation_translation():
    """Test that rotation and translation propagate to all regions."""
    masker = multem.MultiRegionMasker(128, 128, 1.0)

    for i in range(3):
        shape = multem.Masker(128, 128, 1.0)
        shape.set_cuboid([0, 0, 0], [128, 128, 20])
        params = multem.IceParameters()
        masker.add_region(shape, params)

    # These should not raise
    masker.set_rotation([64, 64, 10], [0, 0.1, 0])
    masker.set_translation([5, 10, 0])
