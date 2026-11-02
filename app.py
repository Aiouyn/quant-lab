"""Root application entrypoint for the read-only Desk OS control room (HOST / DESK_PORT env vars; default 127.0.0.1:3000)."""
from __future__ import annotations

import os
from desk.config import SETTINGS
from desk.web.server import serve


def main():
    settings = SETTINGS.ensure()
    port = int(os.getenv('DESK_PORT', os.getenv('DEFAULT_APP_PORT', '3000')))
    host = os.getenv('HOST', '127.0.0.1')
    serve(settings, port=port, host=host)


if __name__ == '__main__':
    main()
