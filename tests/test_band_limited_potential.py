#
# test_band_limited_potential.py
#
# Band-limited atomic potentials (MULTEM_BANDLIMITED_POTENTIAL, which multem turns on at import).
#
# MULTEM's own radial tables lose most of an atom's projected potential at the pixel sizes used for
# cryo-EM, in a share that depends on the element. The band-limited tables deposit all of it, at any
# pixel size and sub-pixel position, and the ice random field follows: its mean is the water
# molecule's integral times the number of molecules, with no correction for the pixel size, and its
# variance is the part of its power spectrum that the band passes. These run on the CPU.
#
import numpy as np
import pytest

import multem


C_POTF = 47.877645145863056  # V A^2, MULTEM's c_Potf

# IcePotentialApproximation's built-in water variance scale and default spectrum (multem_ext.cu)
V0 = 10195.82
DEFAULT_SPECTRUM = dict(a1=0.199, a2=0.801, m1=0.0, m2=1.0 / 2.88, s1=0.731, s2=0.081)


def exact_integral(Z):
    """
    c_Potf f_e(0) for element Z, from the coefficients of the 3D potential multem reports
    (independent of the tables under test): compute_V_params gives
    (cl, cnl) = (c_Potf pi^2 a / b^1.5, 2 pi / sqrt(b)) per Lobato term, and f_e(0) = 2 sum a.

    """
    a_sum = 0.0
    for cl, cnl in multem.compute_V_params("Lobato_0_12", Z, 0):
        if cl == 0 or cnl == 0:
            continue
        b = (2 * np.pi / cnl) ** 2
        a_sum += cl * b**1.5 / (C_POTF * np.pi**2)
    return C_POTF * 2 * a_sum


def water_number_density(density):
    """Molecules per A^3 for a density in g/cm^3, as the ice model computes it."""
    return (density * 1e6 * 1e-30 / 18.01528) * 6.02214076e23


def band_fraction(dR, a1, a2, m1, m2, s1, s2):
    """int P W^2 q dq / int P q dq, by brute-force quadrature (independent of the C++ helper)."""
    q_nyq = 0.5 / dR
    q = np.linspace(0, max(m1, m2) + 15 * max(s1, s2), 2_000_001)
    P = a1 * np.exp(-0.5 * (q - m1) ** 2 / s1**2) + a2 * np.exp(-0.5 * (q - m2) ** 2 / s2**2)
    q1, q2 = 0.5 * q_nyq, q_nyq
    W = np.where(q <= q1, 1.0, 0.5 * (1 + np.cos(np.pi * (q - q1) / (q2 - q1))))
    W = np.where(q >= q2, 0.0, W)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed it
    return trapezoid(P * W**2 * q, q) / trapezoid(P * q, q)


@pytest.fixture
def band_limit(monkeypatch):
    """Select band-limited (True) or MULTEM's own (False) tables; MULTEM reads it per simulation."""

    def select(on):
        monkeypatch.setenv("MULTEM_BANDLIMITED_POTENTIAL", "1" if on else "0")

    return select


def _input(n, dR, lz, dz):
    sc = multem.SystemConfiguration()
    sc.precision = "float"
    sc.device = "host"
    inp = multem.Input()
    inp.simulation_type = "EWRS"
    inp.interaction_model = "Multislice"
    inp.potential_type = "Lobato_0_12"
    inp.potential_slicing = "dz_Proj"
    inp.pn_model = "Still_Atom"
    inp.thick_type = "Whole_Spec"
    inp.bwl = False
    inp.E_0 = 300
    inp.nx = n
    inp.ny = n
    inp.spec_lx = n * dR
    inp.spec_ly = n * dR
    inp.spec_lz = lz
    inp.spec_dz = dz
    inp.iw_x = [n * dR / 2]
    inp.iw_y = [n * dR / 2]
    return sc, inp


def deposited_per_atom(Z, dR, seed, n_side=4, spacing=24.0):
    """Sum of the projected potential over the grid, times the pixel area, per atom (V A^3)."""
    n = int(round(n_side * spacing / dR))
    sc, inp = _input(n, dR, 10.0, 10.0)
    rng = np.random.default_rng(seed)
    atoms = []
    for i in range(n_side):
        for j in range(n_side):
            x = (round((i + 0.5) * spacing / dR) + rng.random()) * dR
            y = (round((j + 0.5) * spacing / dR) + rng.random()) * dR
            atoms.append((Z, x, y, 5.0, 0.0, 1.0, 0, 0))
    inp.spec_atoms = multem.AtomList(atoms)
    slices = []
    multem.compute_projected_potential(
        sc, inp, lambda z0, z1, V: slices.append(np.array(V, dtype=np.float64))
    )
    return np.sum(slices, axis=0).sum() * dR * dR / len(atoms)


def test_on_by_default():
    assert multem.band_limited_potential()


@pytest.mark.parametrize("dR", [0.5, 1.5, 3.0])
@pytest.mark.parametrize("Z", [1, 6, 8])
def test_atoms_keep_their_integral(band_limit, Z, dR):
    band_limit(True)
    ratio = deposited_per_atom(Z, dR, seed=Z) / exact_integral(Z)
    assert abs(ratio - 1) < 2e-3, ratio


def test_original_tables_still_lose_it(band_limit):
    # the fault the band limit fixes, so the test above is not vacuous: oxygen at 1.5 A/px
    band_limit(False)
    ratio = deposited_per_atom(8, 1.5, seed=8) / exact_integral(8)
    assert ratio < 0.5, ratio


def ice_slices(dR, density=0.94, n=128, thickness=30.0, dz=3.0):
    sc, inp = _input(n, dR, thickness, dz)
    # one atom of zero occupancy: the field alone (where atoms are present it is scaled down)
    inp.spec_atoms = multem.AtomList([(1, n * dR / 2, n * dR / 2, thickness / 2, 0.0, 0.0, 0, 0)])
    masker = multem.Masker(n, n, dR)
    ice = multem.IceParameters()
    for k, v in DEFAULT_SPECTRUM.items():
        setattr(ice, k, v)
    ice.density = density
    masker.set_ice_parameters(ice)
    masker.set_cuboid((0, 0, 0), (n * dR, n * dR, thickness))
    slices = []
    multem.compute_projected_potential(
        sc, inp, masker, lambda z0, z1, V: slices.append((z1 - z0, np.array(V, dtype=np.float64)))
    )
    return slices


@pytest.mark.parametrize("dR", [0.5, 1.5, 3.0])
def test_ice_mean_is_the_molecular_integral(band_limit, dR):
    band_limit(True)
    density = 0.94
    m_water = exact_integral(8) + 2 * exact_integral(1)
    slices = ice_slices(dR, density)
    assert len(slices) > 2
    for t, V in slices[1:-1]:
        expected = m_water * water_number_density(density) * t
        assert abs(V.mean() / expected - 1) < 1e-4, (V.mean(), expected)


@pytest.mark.parametrize("dR", [0.5, 1.5, 3.0])
def test_ice_variance_is_the_band_fraction(band_limit, dR):
    band_limit(True)
    density = 0.94
    frac = band_fraction(dR, **DEFAULT_SPECTRUM)
    for t, V in ice_slices(dR, density)[1:-1]:
        expected = V0 * frac * water_number_density(density) * t
        assert abs(V.var() / expected - 1) < 1e-3, (V.var(), expected)


def test_ice_field_carries_the_window(band_limit):
    # nothing at or beyond Nyquist of the band: the field's power outside the disc |q| < q_nyq
    band_limit(True)
    dR = 1.5
    slices = ice_slices(dR)
    _, V = slices[len(slices) // 2]
    F = np.abs(np.fft.fft2(V - V.mean())) ** 2
    n = V.shape[0]
    q = np.hypot(*np.meshgrid(np.fft.fftfreq(n, dR), np.fft.fftfreq(n, dR), indexing="ij"))
    outside = F[q >= 0.5 / dR].sum() / F.sum()
    assert outside < 1e-6, outside
