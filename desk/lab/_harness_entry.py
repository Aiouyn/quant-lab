"""Jailed entry point for the trusted harness (`run_job`): same as `python -m desk.lab.harness job.json out.json`."""
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])   # repo root (desk/ is the only part mounted), not desk/lab
from desk.lab.harness import main

main()
