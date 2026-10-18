"""Root application entrypoint for Desk OS v3 Control Room & RSI Engine on 0.0.0.0:3000."""
from __future__ import annotations

import os
from desk.config import SETTINGS
from desk.web.server import serve


def main():
    settings = SETTINGS.ensure()
    port = int(os.getenv('DESK_PORT', os.getenv('DEFAULT_APP_PORT', '3000')))
    host = os.getenv('HOST', '0.0.0.0')
    serve(settings, port=port, host=host)


if __name__ == '__main__':
    main()
