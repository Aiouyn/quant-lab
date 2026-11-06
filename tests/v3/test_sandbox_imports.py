"""Modules agents load inside the sandbox must not import network clients the guard blocks (2026-11-06: CarryStore
pulled httpx through desk.data.carry_data, so carry_panel() failed in every sandbox)."""
import subprocess
import sys


def test_carry_store_does_not_import_httpx():
    code = ('import sys; import desk.lab.carry, desk.data.carry_data; '
            'print("httpx" in sys.modules)')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == 'False'

