"""Launch the evidence dashboard. Importing this module starts no work."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

def run_server():
    from quantdesk.cli import main
    return main(['serve'])

if __name__=='__main__':
    raise SystemExit(run_server())
