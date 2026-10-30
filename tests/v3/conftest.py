"""Shared pytest config for tests/v3.

`needs_real_store` marks integration tests that build the desk Parquet stores from real
remote sources (Yahoo Finance, Hyperliquid, Binance, Polymarket) and from the local
Hyperliquid asset-ctx archive at `desk.data.perps_panel.HLCTX`. Since #12 the store builder
has no synthetic fallback, so these tests cannot run offline or in CI. They also run the full
Truth Machine (dev -> val -> one-shot holdout) on whatever data was fetched, so they are
opt-in: set DESK_REAL_STORE_TESTS=1 on a machine that has network access and the HLCTX files.
"""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get('DESK_REAL_STORE_TESTS') == '1':
        return
    skip = pytest.mark.skip(reason='needs real desk stores (network + HL asset-ctx files at '
                                   'perps_panel.HLCTX); no synthetic fallback since #12. '
                                   'Opt in with DESK_REAL_STORE_TESTS=1.')
    for item in items:
        if 'needs_real_store' in item.keywords:
            item.add_marker(skip)
