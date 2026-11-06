import numpy as np
import pytest

import multem


def columns(n, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        element=rng.integers(1, 80, n).astype(np.uint8),
        x=rng.uniform(0, 1000, n).astype(np.float32),
        y=rng.uniform(0, 1000, n).astype(np.float32),
        z=rng.uniform(0, 500, n).astype(np.float32),
        sigma=rng.uniform(0, 0.2, n).astype(np.float32),
        occupancy=np.ones(n, np.float32),
        region=np.zeros(n, np.int32),
        charge=rng.integers(-2, 3, n).astype(np.int8),
    )


def test_same_atoms_as_tuples():
    c = columns(1000)
    ref = multem.AtomList(zip(*(c[k].tolist() for k in
                                ("element", "x", "y", "z", "sigma", "occupancy", "region", "charge"))))
    got = multem.AtomList.from_arrays(**c)
    assert isinstance(got, multem.AtomList)
    assert len(got) == len(ref) == 1000
    assert all(a == b for a, b in zip(ref, got))


def test_float64_and_noncontiguous_columns():
    c = columns(50, seed=1)
    c["x"] = c["x"].astype(np.float64)[::-1].copy()[::-1]
    c["y"] = np.repeat(c["y"], 2)[::2]          # a strided view
    got = multem.AtomList.from_arrays(**c)
    assert [a[2] for a in got] == [float(v) for v in c["y"]]


def test_empty():
    assert len(multem.AtomList.from_arrays(**columns(0))) == 0


def test_assignable_to_input():
    inp = multem.Input()
    inp.spec_atoms = multem.AtomList.from_arrays(**columns(10))
    assert len(inp.spec_atoms) == 10


def test_length_mismatch_is_an_error():
    c = columns(10)
    c["z"] = c["z"][:9]
    with pytest.raises(ValueError, match="'z' has 9 entries, expected 10"):
        multem.AtomList.from_arrays(**c)


def test_float_integer_column_is_an_error():
    c = columns(10)
    c["element"] = c["element"].astype(np.float32)   # would truncate silently under a forced cast
    with pytest.raises(TypeError, match="'element'"):
        multem.AtomList.from_arrays(**c)


def test_two_dimensional_column_is_an_error():
    c = columns(10)
    c["x"] = c["x"].reshape(2, 5)
    with pytest.raises(ValueError, match="one-dimensional"):
        multem.AtomList.from_arrays(**c)
