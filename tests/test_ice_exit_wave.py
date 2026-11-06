#
# test_ice_exit_wave.py
#
# The ice model must reach the exit wave on every path. The GPU multislice once
# took a pipelined slice loop that computed the atomic potential directly and
# skipped the potential function (the ice) and the static B factor, so a 50 nm
# ice slab left a plane wave on the GPU (phase s.d. 6e-5 rad against 0.09) while the CPU path and the projected
# potential showed it. These tests render a pure-ice slab (two light anchor atoms
# set the slicing) and check that the exit-wave phase carries the ice, on the
# host and, when there is one, on the GPU.
#
import numpy as np
import pytest

import multem

N = 256  # pixels per side
DR = 1.0  # A per pixel
LZ = 500.0  # slab thickness, A
DZ = 3.0  # slice thickness, A


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
    # two hydrogen anchors, at the top and bottom of the slab, so that slices exist
    inp.spec_atoms = multem.AtomList(
        [(1, 10.0, 10.0, 1.0, 0.0, 1.0, 0, 0), (1, 20.0, 20.0, LZ - 1.0, 0.0, 1.0, 0, 0)]
    )
    return inp


def _masker():
    masker = multem.Masker(N, N, DR)
    ice = multem.IceParameters()
    ice.m1, ice.m2 = 0.0, 1 / 2.88
    ice.s1, ice.s2 = 0.731, 0.081
    ice.a1, ice.a2 = 0.199, 0.801
    ice.density = 0.94
    masker.set_ice_parameters(ice)
    masker.set_cuboid((0.0, 0.0, 0.0), (N * DR, N * DR, LZ))
    return masker


def _phase_sd(device, ice):
    sc = multem.SystemConfiguration()
    sc.precision = "float"
    sc.device = device
    inp = _input()
    out = multem.simulate(sc, inp, _masker()) if ice else multem.simulate(sc, inp)
    psi = np.array(out.data[0].psi_coh)
    ref = psi.mean()
    ref = ref / abs(ref) if abs(ref) > 0 else 1.0
    return float(np.std(np.angle(psi * np.conj(ref))))


def test_ice_reaches_the_exit_wave_on_the_host():
    with_ice = _phase_sd("host", True)
    without = _phase_sd("host", False)
    # 50 nm of ice filling the field: a phase texture of ~0.09 rad (measured); the two anchor
    # atoms alone, far less. The broken GPU loop gave 6e-5 rad with ice.
    assert with_ice > 0.03, with_ice
    assert with_ice > 20 * without, (with_ice, without)


@pytest.mark.skipif(not multem.is_gpu_available(), reason="no GPU")
def test_ice_reaches_the_exit_wave_on_the_gpu():
    gpu = _phase_sd("device", True)
    host = _phase_sd("host", True)
    without = _phase_sd("device", False)
    assert gpu > 0.03, gpu
    assert gpu > 20 * without, (gpu, without)
    # independent random fields, so not equal; the same statistics within 20%
    assert abs(gpu / host - 1) < 0.2, (gpu, host)
