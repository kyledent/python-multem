#
# test_mask_pixel_grid.py
#
# Region masks sample the same grid points as MULTEM's atom potential: pixel i is at x = i * dx (its
# corner), not at its centre (i + 0.5) * dx. With centres, every region sat half a pixel away from the
# atoms in the same render, in x and in y.
#
import numpy as np
import pytest

import multem

N = 64
DR = 1.5


def devices():
    return ["host"] + (["device"] if multem.is_gpu_available() else [])


def masks(m, zs, ze):
    out = {"loop": np.array(m.compute(zs, ze)).reshape(N, N)}
    for d in devices():
        out[d] = np.array(multem.compute_mask_engine(m, zs, ze, d == "device")).reshape(N, N)
    return out


def test_box_face_switches_on_the_corner_grid():
    # a face at x = 10.25 px: pixel 10 (x = 10 px) is outside, pixel 11 inside; with centres, pixel
    # 10 (x = 10.5 px) was inside
    m = multem.Masker(N, N, DR)
    m.set_cuboid((10.25 * DR, 0.0, 0.0), (20 * DR, N * DR, 30.0))
    for name, mask in masks(m, 10.0, 13.0).items():
        col = mask[:, N // 2]
        assert col[10] == 0 and col[11] == 1, (name, col[8:14])
        assert col[30] == 1 and col[31] == 0, (name, col[28:33])   # far face at 30.25 px


def test_sphere_on_the_corner_grid():
    c = (20 * DR + 0.3, 30 * DR - 0.4, 15.0)
    r = 7.3
    m = multem.Masker(N, N, DR)
    m.set_sphere(list(c), r)
    zs, ze = 13.5, 16.5
    dz = 0.5 * (zs + ze) - c[2]
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    ref = ((i * DR - c[0]) ** 2 + (j * DR - c[1]) ** 2 + dz ** 2 <= r * r).astype(int)
    for name, mask in masks(m, zs, ze).items():
        assert np.array_equal(mask, ref), name


@pytest.mark.parametrize("device", devices())
def test_masks_and_atoms_share_a_grid(device):
    # one heavy atom exactly on grid point (20, 30); a small sphere mask centred on it must be centred
    # on the same pixel as the atom's projected potential
    x0, y0 = 20 * DR, 30 * DR
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
    inp.spec_lx = inp.spec_ly = N * DR
    inp.spec_lz = 30.0
    inp.spec_dz = 30.0
    inp.iw_x = [N * DR / 2]
    inp.iw_y = [N * DR / 2]
    inp.spec_atoms = multem.AtomList([(79, x0, y0, 15.0, 0.0, 1.0, 0, 0)])
    sc = multem.SystemConfiguration()
    sc.precision = "double"
    sc.device = device
    V = []
    multem.compute_projected_potential(sc, inp, lambda z0, z1, v: V.append(np.array(v)))
    P = np.sum(V, axis=0)
    m = multem.Masker(N, N, DR)
    m.set_sphere([x0, y0, 15.0], 1.6)
    mask = np.array(multem.compute_mask_engine(m, 14.0, 16.0, device == "device")).reshape(P.shape)
    peak = np.unravel_index(int(np.argmax(P)), P.shape)
    centre = tuple(int(round(v)) for v in np.argwhere(mask == 1).mean(axis=0))
    print(f"\n{device}: atom peak at {peak}, sphere mask centred at {centre}, {int(mask.sum())} px")
    assert peak == centre
