#
# test_ice_cost_f3.py
#
# The band-limited potential must not make a masked simulation slower. It
# once did, by ~1.5 ms a slice: the band-limited variance fraction was
# recomputed for every slice, rebuilding two 256-point Gauss-Legendre rules
# each time, although it depends only on the ice's power spectrum and the
# grid spacing. It ran through the masker even with no ice in it, so the
# comparison is F3 on against F3 off, with ice and without. The timings are
# printed (run with -s).
#
import os
import time

import numpy as np
import pytest

import multem

N = 256
DR = 1.5
LZ = 500.0
DZ = 1.0
REPEATS = 3


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
    inp.spec_atoms = multem.AtomList(
        [(1, 10.0, 10.0, 1.0, 0.0, 1.0, 0, 0), (1, 20.0, 20.0, LZ - 1.0, 0.0, 1.0, 0, 0)]
    )
    return inp


def _masker(ice):
    masker = multem.Masker(N, N, DR)
    if ice:
        p = multem.IceParameters()
        p.m1, p.m2 = 0.0, 1 / 2.88
        p.s1, p.s2 = 0.731, 0.081
        p.a1, p.a2 = 0.199, 0.801
        p.density = 0.94
        masker.set_ice_parameters(p)
        masker.set_cuboid((0.0, 0.0, 0.0), (N * DR, N * DR, LZ))
        masker.set_seed(5)
    return masker


def _time(device, ice, f3):
    os.environ["MULTEM_BANDLIMITED_POTENTIAL"] = "1" if f3 else "0"
    sc = multem.SystemConfiguration()
    sc.precision = "float"
    sc.device = device
    best = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = multem.simulate(sc, _input(), _masker(ice))
        dt = time.perf_counter() - t0
        assert np.isfinite(np.abs(np.array(out.data[0].psi_coh))).all()
        best = dt if best is None else min(best, dt)
    return best


@pytest.mark.parametrize(
    "device",
    ["host", pytest.param("device", marks=pytest.mark.skipif(
        not multem.is_gpu_available(), reason="no GPU"))],
)
def test_ice_costs_the_same_with_f3(device):
    saved = os.environ.get("MULTEM_BANDLIMITED_POTENTIAL")
    try:
        t = {(ice, f3): _time(device, ice, f3) for ice in (False, True) for f3 in (False, True)}
    finally:
        if saved is None:
            os.environ.pop("MULTEM_BANDLIMITED_POTENTIAL", None)
        else:
            os.environ["MULTEM_BANDLIMITED_POTENTIAL"] = saved
    ice_off = t[(True, False)] - t[(False, False)]
    ice_on = t[(True, True)] - t[(False, True)]
    n_slices = int(LZ / DZ)
    print(
        f"\n{device}: {n_slices} slices; no ice F3 0/1 {t[(False, False)]:.3f}/{t[(False, True)]:.3f} s; "
        f"ice F3 0/1 {t[(True, False)]:.3f}/{t[(True, True)]:.3f} s; "
        f"ice cost F3 0 {ice_off:.3f} s, F3 1 {ice_on:.3f} s "
        f"({1e3 * (ice_on - ice_off) / n_slices:.3f} ms/slice extra)"
    )
    # a per-slice recomputation costs ~1.5 ms a slice, ~0.7 s here
    for ice in (False, True):
        extra = t[(ice, True)] - t[(ice, False)]
        assert extra < 0.1 + 0.1 * t[(ice, False)], (ice, extra, t)
