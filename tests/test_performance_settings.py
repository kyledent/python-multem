#
# test_performance_settings.py
#
# The host-side performance settings multem applies on import. They are applied at import time,
# so each case imports multem in a fresh interpreter with a controlled environment.
#
import json
import os
import re
import subprocess
import sys

import pytest

import multem


# A small still-atom exit-wave simulation on the CPU (no GPU needed). 42^3 = 74,088 atoms: above
# the size at which MULTEM's parallel sort takes over from the sequential one.
_SIMULATE = r"""
import multem
system_conf = multem.SystemConfiguration()
system_conf.device = "host"
system_conf.precision = "float"
inp = multem.Input()
inp.simulation_type = "EWRS"
inp.interaction_model = "Multislice"
inp.potential_type = "Lobato_0_12"
inp.potential_slicing = "dz_Proj"
inp.pn_model = "Still_Atom"
inp.thick_type = "Whole_Spec"
inp.E_0 = 300.0
n, a = 42, 2.0
inp.spec_atoms = multem.AtomList(
    [(14, a * i + 1, a * j + 1, a * k + 1, 0.085, 1, 0, 0)
     for i in range(n) for j in range(n) for k in range(n)])
inp.spec_lx = inp.spec_ly = inp.spec_lz = a * n + 2
inp.spec_dz = 2.0
inp.nx = inp.ny = 64
multem.simulate(system_conf, inp)
"""


def _environment_after_import(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MULTEM_")}
    env.update(extra)
    code = (
        "import json, os, multem; "
        "print(json.dumps({k: v for k, v in os.environ.items() if k.startswith('MULTEM_')}))"
    )
    out = subprocess.check_output([sys.executable, "-W", "ignore", "-c", code], env=env, text=True)
    return json.loads(out.strip().splitlines()[-1])


def test_optimisations_on_and_threads_default():
    got = _environment_after_import({})
    for name in multem.PERFORMANCE_TOGGLES:
        assert got[name] == "1", name
    assert got["MULTEM_NUM_THREADS"] == str(multem.default_num_threads())


def test_user_settings_win():
    got = _environment_after_import({"MULTEM_PAR_SORT": "0", "MULTEM_NUM_THREADS": "3"})
    assert got["MULTEM_PAR_SORT"] == "0"
    assert got["MULTEM_NUM_THREADS"] == "3"
    assert got["MULTEM_PAR_COPY"] == "1"


def test_invalid_thread_count_is_replaced():
    got = _environment_after_import({"MULTEM_NUM_THREADS": "lots"})
    assert got["MULTEM_NUM_THREADS"] == str(multem.default_num_threads())


def test_default_is_75_percent_of_available_cpus():
    assert multem.default_num_threads() == max(1, int(multem.available_cpus() * 0.75))
    assert multem.default_num_threads(fraction=0.01) == 1
    assert 1 <= multem.available_cpus() <= (os.cpu_count() or 1)


def test_settings_reach_multem():
    # End to end: the settings applied on import must be the ones MULTEM's C++ actually uses.
    # MULTEM_SORT_CHECK=1 makes it sort both ways and print how many threads it used and how many
    # atoms differ from the sequential sort -- which must be none.
    env = {k: v for k, v in os.environ.items() if not k.startswith("MULTEM_")}
    env["MULTEM_SORT_CHECK"] = "1"
    run = subprocess.run(
        [sys.executable, "-c", _SIMULATE], env=env, capture_output=True, text=True, timeout=900
    )
    assert run.returncode == 0, run.stderr[-2000:]
    checks = [line for line in run.stderr.splitlines() if "[check] sort_by_z" in line]
    assert checks, "MULTEM printed no sort check -- is the MULTEM submodule the fork?\n" + run.stderr[-2000:]
    m = re.search(r"(\d+) atoms, (\d+) threads, (\d+) differ", checks[0])
    assert m, checks[0]
    assert int(m.group(1)) == 42 ** 3
    assert int(m.group(2)) == multem.default_num_threads()
    assert int(m.group(3)) == 0


def test_set_num_threads():
    before = multem.get_num_threads()
    try:
        multem.set_num_threads(5)
        assert multem.get_num_threads() == 5
        with pytest.raises(ValueError):
            multem.set_num_threads(0)
    finally:
        multem.set_num_threads(before)
