"""In-jail boot step: set per-namespace rlimits, then run the target script as __main__."""
import resource
import runpy
import sys

nproc, nofile = int(sys.argv[1]), int(sys.argv[2])
resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
script = sys.argv[3]
sys.argv = sys.argv[3:]
sys.path[0] = str(__import__('pathlib').Path(script).resolve().parent)
runpy.run_path(script, run_name='__main__')
