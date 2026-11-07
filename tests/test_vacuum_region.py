#
# test_vacuum_region.py
#
# A MultiRegionMasker region with density 0 is vacuum: it adds no random field,
# and other regions can still subtract its shape. Here an ice slab subtracts a
# vacuum cylinder through it, so the hole must look like no ice and the rest
# like ice.
#
import numpy as np
import pytest

import multem

N = 128
DR = 1.5
LZ = 300.0
DZ = 3.0
HOLE_R = 30.0
# two hydrogen anchors at the top and bottom, so that slices exist
ANCHORS = [(1, 10.0, 10.0, 1.0, 0.0, 1.0, 0, 0), (1, 20.0, 20.0, LZ - 1.0, 0.0, 1.0, 0, 0)]


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
    p.m1, p.m2 = 0.0, 1 / 2.88
    p.s1, p.s2 = 0.731, 0.081
    p.a1, p.a2 = 0.199, 0.801
    p.density = density
    return p


def _slab():
    shape = multem.Masker(N, N, DR)
    shape.set_cuboid((0.0, 0.0, 0.0), (N * DR, N * DR, LZ))
    return shape


def _hole():
    c = N * DR / 2
    shape = multem.Masker(N, N, DR)
    shape.set_cylinder([c, c, 0.0], [0, 0, 1], LZ, [HOLE_R], [0], [0])
    return shape


def _phase(masker, device="host"):
    sc = multem.SystemConfiguration()
    sc.precision = "float"
    sc.device = device
    psi = np.array(multem.simulate(sc, _input(), masker).data[0].psi_coh)
    return np.angle(psi)


def _radius():
    y, x = np.mgrid[0:N, 0:N]
    c = N / 2
    return np.hypot(x + 0.5 - c, y + 0.5 - c) * DR


def test_probe():
    assert multem.MultiRegionMasker.supports_vacuum_regions is True


def test_vacuum_alone_is_no_ice():
    # A multi-region simulation adds two hydrogen sentinels at (0, 0, zmin) and
    # (0, 0, zmax) of the masker to fix the slicing; the reference without a
    # masker carries the same two atoms, so only the regions can differ.
    m = multem.MultiRegionMasker(N, N, DR)
    m.add_region(_hole(), _ice(0.0))
    with_vacuum = _phase(m)
    inp = _input()
    sentinels = [(1, 0.0, 0.0, m.zmin(), 0.0, 1.0, 0, 0), (1, 0.0, 0.0, m.zmax(), 0.0, 1.0, 0, 0)]
    inp.spec_atoms = multem.AtomList(ANCHORS + sentinels)
    sc = multem.SystemConfiguration()
    sc.precision = "float"
    sc.device = "host"
    plain = np.angle(np.array(multem.simulate(sc, inp).data[0].psi_coh))
    diff = float(np.max(np.abs(with_vacuum - plain)))
    print(f"\nvacuum alone vs no masker with the same sentinels: max |dphase| {diff:.3g} rad")
    assert diff < 1e-5


@pytest.mark.parametrize(
    "device",
    ["host", pytest.param("device", marks=pytest.mark.skipif(
        not multem.is_gpu_available(), reason="no GPU"))],
)
def test_ice_slab_minus_vacuum_hole(device):
    m = multem.MultiRegionMasker(N, N, DR)
    m.add_region(_hole(), _ice(0.0))        # region 0: vacuum
    m.add_region(_slab(), _ice(), [0])      # region 1: ice, minus the hole
    m.set_seed(3)
    ph = _phase(m, device)
    r = _radius()
    inside = ph[r < HOLE_R - 6 * DR]
    outside = ph[r > HOLE_R + 6 * DR]
    s_in, s_out = float(inside.std()), float(outside.std())
    m_in, m_out = float(inside.mean()), float(outside.mean())
    print(f"\n{device}: phase s.d. inside {s_in:.4g}, outside {s_out:.4g}; mean inside {m_in:.4g}, outside {m_out:.4g} rad")
    # ice gives a mean phase and texture; the hole gives (nearly) neither
    assert s_out > 0.03
    assert s_in < 0.2 * s_out
    assert m_out - m_in > 0.5
