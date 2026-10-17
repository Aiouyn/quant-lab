from .cli import main
import sys

try:
    raise SystemExit(main())
except Exception as error:
    print(f'Quant Lab: {type(error).__name__}: {error}',file=sys.stderr)
    raise SystemExit(1)
