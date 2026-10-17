"""
High-Performance OKX 5000-Level L2 Orderbook Reconstructor & Feature Generator
Extracts microsecond/millisecond-level depth vectors, order flow imbalances, and forward metrics.
"""

import tarfile
import json
import numpy as np
import polars as pl
from typing import Dict, Any, List, Optional
from pathlib import Path

class OrderbookReconstructor:
    __slots__ = ('bids', 'asks', 'prev_bid_p0', 'prev_bid_v0', 'prev_ask_p0', 'prev_ask_v0')

    def __init__(self):
        self.bids: Dict[float, tuple] = {}  # price -> (volume, order_count)
        self.asks: Dict[float, tuple] = {}
        self.prev_bid_p0 = 0.0
        self.prev_bid_v0 = 0.0
        self.prev_ask_p0 = 0.0
        self.prev_ask_v0 = 0.0

    def apply_snapshot(self, bids_raw: list, asks_raw: list):
        self.bids.clear()
        self.asks.clear()
        for p_str, v_str, n_str in bids_raw:
            v = float(v_str)
            if v > 0:
                self.bids[float(p_str)] = (v, int(n_str))
        for p_str, v_str, n_str in asks_raw:
            v = float(v_str)
            if v > 0:
                self.asks[float(p_str)] = (v, int(n_str))

    def apply_update(self, bids_raw: list, asks_raw: list):
        for p_str, v_str, n_str in bids_raw:
            p = float(p_str)
            v = float(v_str)
            if v == 0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = (v, int(n_str))

        for p_str, v_str, n_str in asks_raw:
            p = float(p_str)
            v = float(v_str)
            if v == 0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = (v, int(n_str))

    def extract_state(self, ts_ms: int) -> Optional[Dict[str, Any]]:
        if not self.bids or not self.asks:
            return None

        sorted_bids = sorted(self.bids.keys(), reverse=True)
        sorted_asks = sorted(self.asks.keys())

        if not sorted_bids or not sorted_asks:
            return None

        bid_p0 = sorted_bids[0]
        ask_p0 = sorted_asks[0]

        # Inverted book protection
        if bid_p0 >= ask_p0:
            return None

        bid_v0, bid_n0 = self.bids[bid_p0]
        ask_v0, ask_n0 = self.asks[ask_p0]

        spread = ask_p0 - bid_p0
        mid_p = (ask_p0 + bid_p0) / 2.0

        # Micro-price (Volume-weighted fair value)
        denom_0 = bid_v0 + ask_v0
        micro_p = (bid_v0 * ask_p0 + ask_v0 * bid_p0) / denom_0 if denom_0 > 0 else mid_p
        micro_dev = micro_p - mid_p

        # Top 1, 5, 20 cumulative depths
        bid_v5 = sum(self.bids[p][0] for p in sorted_bids[:5])
        ask_v5 = sum(self.asks[p][0] for p in sorted_asks[:5])
        bid_n5 = sum(self.bids[p][1] for p in sorted_bids[:5])
        ask_n5 = sum(self.asks[p][1] for p in sorted_asks[:5])

        bid_v20 = sum(self.bids[p][0] for p in sorted_bids[:20])
        ask_v20 = sum(self.asks[p][0] for p in sorted_asks[:20])

        qi_1 = (bid_v0 - ask_v0) / denom_0 if denom_0 > 0 else 0.0
        qi_5 = (bid_v5 - ask_v5) / (bid_v5 + ask_v5) if (bid_v5 + ask_v5) > 0 else 0.0
        qi_20 = (bid_v20 - ask_v20) / (bid_v20 + ask_v20) if (bid_v20 + ask_v20) > 0 else 0.0

        # Order Count Skew at Top 5
        total_n5 = bid_n5 + ask_n5
        count_skew_5 = (bid_n5 - ask_n5) / total_n5 if total_n5 > 0 else 0.0

        # Multi-Level Order Flow Imbalance (OFI) (Cont-Kukanov-Stoikov)
        if self.prev_bid_p0 > 0:
            if bid_p0 > self.prev_bid_p0:
                delta_bid = bid_v0
            elif bid_p0 == self.prev_bid_p0:
                delta_bid = bid_v0 - self.prev_bid_v0
            else:
                delta_bid = 0.0

            if ask_p0 < self.prev_ask_p0:
                delta_ask = ask_v0
            elif ask_p0 == self.prev_ask_p0:
                delta_ask = ask_v0 - self.prev_ask_v0
            else:
                delta_ask = 0.0

            ofi = delta_bid - delta_ask
        else:
            ofi = 0.0

        self.prev_bid_p0 = bid_p0
        self.prev_bid_v0 = bid_v0
        self.prev_ask_p0 = ask_p0
        self.prev_ask_v0 = ask_v0

        return {
            'ts_ms': ts_ms,
            'bid_p0': bid_p0,
            'bid_v0': bid_v0,
            'bid_n0': bid_n0,
            'ask_p0': ask_p0,
            'ask_v0': ask_v0,
            'ask_n0': ask_n0,
            'spread': spread,
            'mid_price': mid_p,
            'micro_price': micro_p,
            'micro_dev': micro_dev,
            'bid_v5': bid_v5,
            'ask_v5': ask_v5,
            'qi_1': qi_1,
            'qi_5': qi_5,
            'qi_20': qi_20,
            'count_skew_5': count_skew_5,
            'ofi': ofi
        }

def load_and_extract_features(tar_path: str, max_rows: Optional[int] = None) -> pl.DataFrame:
    """Parses an OKX 5000lv .tar.gz archive and returns a rich Polars DataFrame."""
    book = OrderbookReconstructor()
    records: List[Dict[str, Any]] = []

    with tarfile.open(tar_path, 'r:gz') as tar:
        member = tar.getmembers()[0]
        f = tar.extractfile(member)
        count = 0

        for line in f:
            obj = json.loads(line)
            action = obj.get('action')
            ts = int(obj.get('ts'))
            bids = obj.get('bids', [])
            asks = obj.get('asks', [])

            if action == 'snapshot':
                book.apply_snapshot(bids, asks)
            elif action == 'update':
                book.apply_update(bids, asks)

            state = book.extract_state(ts)
            if state is not None:
                records.append(state)
                count += 1
                if max_rows and count >= max_rows:
                    break

    df = pl.DataFrame(records)

    # Add forward returns (basis points) for horizons: 1s, 5s, 10s, 30s, 60s
    mid = df['mid_price']
    df = df.with_columns([
        ((mid.shift(-1) - mid) / mid * 10000.0).alias('fwd_ret_1s_bps'),
        ((mid.shift(-5) - mid) / mid * 10000.0).alias('fwd_ret_5s_bps'),
        ((mid.shift(-10) - mid) / mid * 10000.0).alias('fwd_ret_10s_bps'),
        ((mid.shift(-30) - mid) / mid * 10000.0).alias('fwd_ret_30s_bps'),
        ((mid.shift(-60) - mid) / mid * 10000.0).alias('fwd_ret_60s_bps'),
        # Realized volatility (rolling 30-tick return std in bps)
        (mid.pct_change() * 10000.0).rolling_std(window_size=30).alias('vol_30s_bps')
    ])

    return df
