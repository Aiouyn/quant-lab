"""Full-depth OKX L2 reconstruction on a dense tick grid, with per-second deep-book features.

Key correction vs v2: OKX archives contain a full 5000-level snapshot every 15 minutes on an
otherwise continuous 1 Hz diff stream. A periodic snapshot is a *resync*, not a break. We measure
how far the reconstructed book had drifted (quality metric) and keep the series continuous.
Breaks only happen on genuine timestamp gaps (> max_gap_ms) which mark rows invalid.
"""
from __future__ import annotations

import math
import numpy as np

TICK = 0.1                     # BTC-USDT-SWAP tick size
BANDS_BPS = (2, 5, 10, 25, 50, 100)
SWEEP_CONTRACTS = (100, 1000, 5000)   # 1, 10, 50 BTC
TOP = 10

COLUMNS = (
    ['bid_px', 'ask_px', 'mid', 'spread_bps', 'micro_dev_bps',
     'bid_sz0', 'ask_sz0', 'bid_n0', 'ask_n0', 'qi1', 'qi5', 'qi10']
    + [f'{s}dep_{d}' for d in BANDS_BPS for s in ('b', 'a')]
    + ['bcnt_10', 'acnt_10', 'bcnt_50', 'acnt_50', 'ofi1', 'ofi5', 'ofi10',
       'bwall_sz', 'bwall_bps', 'awall_sz', 'awall_bps']
    + [f'{s}imp_{q // 100}' for q in SWEEP_CONTRACTS for s in ('b', 'a')]
)


def parse_levels(levels):
    if not levels:
        return None
    a = np.array(levels, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError('malformed levels')
    idx = np.rint(a[:, 0] / TICK).astype(np.int64)
    size = a[:, 1].astype(np.float32)
    cnt = a[:, 2].astype(np.int32) if a.shape[1] >= 3 else np.zeros(len(a), np.int32)
    return idx, size, cnt


class GridBook:
    def __init__(self, max_price: float = 400_000.0):
        n = int(max_price / TICK) + 2
        self.n = n
        self.bid = np.zeros(n, np.float32)
        self.ask = np.zeros(n, np.float32)
        self.bidn = np.zeros(n, np.int32)
        self.askn = np.zeros(n, np.int32)
        self.bb = -1
        self.ba = -1
        self.ready = False
        self.prev_top = None
        self.q = {'messages': 0, 'snapshots': 0, 'resyncs': 0, 'resync_top20_mismatch_sum': 0.0,
                  'resync_mid_jump_bps_max': 0.0, 'crossed': 0, 'invalid_rows': 0}

    # ---------------------------------------------------------------- updates
    def _set(self, side, levels):
        parsed = parse_levels(levels)
        if parsed is None:
            return None
        idx, size, cnt = parsed
        if idx.min() < 1 or idx.max() >= self.n - 1:
            raise ValueError('price outside grid')
        if side == 'b':
            self.bid[idx] = size; self.bidn[idx] = cnt
        else:
            self.ask[idx] = size; self.askn[idx] = cnt
        return idx, size

    def _top_levels(self, k=20):
        if self.bb < 0 or self.ba < 0:
            return None
        lo = max(0, self.bb - 5000)
        nzb = np.flatnonzero(self.bid[lo:self.bb + 1])[::-1][:k] + lo
        nza = np.flatnonzero(self.ask[self.ba:self.ba + 5000])[:k] + self.ba
        return nzb, self.bid[nzb].copy(), nza, self.ask[nza].copy()

    def apply(self, msg) -> bool:
        """Apply one archive message. Returns True when the book is usable afterwards."""
        self.q['messages'] += 1
        action = msg.get('action')
        if action == 'snapshot':
            before = self._top_levels(20) if self.ready else None
            mid_before = self.mid() if self.ready else None
            for arr in (self.bid, self.ask, self.bidn, self.askn):
                arr.fill(0)
            b = self._set('b', msg.get('bids'))
            a = self._set('a', msg.get('asks'))
            if b is None or a is None:
                self.ready = False
                return False
            nzb = np.flatnonzero(self.bid); nza = np.flatnonzero(self.ask)
            self.bb = int(nzb[-1]) if len(nzb) else -1
            self.ba = int(nza[0]) if len(nza) else -1
            self.ready = self.bb > 0 and self.ba > 0
            self.q['snapshots'] += 1
            if before is not None and self.ready:
                after = self._top_levels(20)
                self.q['resyncs'] += 1
                mism = 0.0
                for (pi, si), (pj, sj) in (((before[0], before[1]), (after[0], after[1])),
                                           ((before[2], before[3]), (after[2], after[3]))):
                    m = min(len(pi), len(pj))
                    same = len(np.intersect1d(pi[:m], pj[:m]))  # price-level presence (sizes move every second)
                    mism += 1 - same / max(m, 1)
                self.q['resync_top20_mismatch_sum'] += mism / 2
                jump = abs(self.mid() / mid_before - 1) * 1e4
                self.q['resync_mid_jump_bps_max'] = max(self.q['resync_mid_jump_bps_max'], jump)
            return self.ready
        if action != 'update' or not self.ready:
            return False
        b = self._set('b', msg.get('bids'))
        a = self._set('a', msg.get('asks'))
        if b is not None:
            idx, size = b
            live = idx[size > 0]
            if len(live) and live.max() > self.bb:
                self.bb = int(live.max())
            elif self.bid[self.bb] == 0:
                self.bb = self._search(self.bid, self.bb, -1)
        if a is not None:
            idx, size = a
            live = idx[size > 0]
            if len(live) and live.min() < self.ba:
                self.ba = int(live.min())
            elif self.ask[self.ba] == 0:
                self.ba = self._search(self.ask, self.ba, 1)
        if self.bb < 0 or self.ba < 0:
            self.ready = False
            return False
        return True

    @staticmethod
    def _search(arr, start, direction):
        w = 256
        n = len(arr)
        while w < n:
            if direction < 0:
                lo = max(0, start - w)
                nz = np.flatnonzero(arr[lo:start])
                if len(nz):
                    return int(lo + nz[-1])
                if lo == 0:
                    return -1
            else:
                hi = min(n, start + 1 + w)
                nz = np.flatnonzero(arr[start + 1:hi])
                if len(nz):
                    return int(start + 1 + nz[0])
                if hi == n:
                    return -1
            w *= 4
        return -1

    def mid(self):
        return (self.bb + self.ba) * TICK / 2

    # --------------------------------------------------------------- features
    def features(self):
        """Deep-book state at the current instant (all causal; uses only the current book and the
        previous top-of-book snapshot for OFI)."""
        bb, ba = self.bb, self.ba
        if bb >= ba:
            self.q['crossed'] += 1
            self.prev_top = None
            return None
        bp, ap = bb * TICK, ba * TICK
        mid = (bp + ap) / 2
        K = int(math.ceil(mid * BANDS_BPS[-1] / 1e4 / TICK)) + 2
        bseg = self.bid[max(0, bb - K + 1):bb + 1][::-1].astype(np.float64)
        aseg = self.ask[ba:ba + K].astype(np.float64)
        bcn = self.bidn[max(0, bb - K + 1):bb + 1][::-1].astype(np.float64)
        acn = self.askn[ba:ba + K].astype(np.float64)
        cb, ca = np.cumsum(bseg), np.cumsum(aseg)
        ccb, cca = np.cumsum(bcn), np.cumsum(acn)
        half = (ap - bp) / 2
        out = {}
        bsz0, asz0 = float(bseg[0]), float(aseg[0])
        out['bid_px'] = bp; out['ask_px'] = ap; out['mid'] = mid
        out['spread_bps'] = (ap - bp) / mid * 1e4
        micro = (ap * bsz0 + bp * asz0) / max(bsz0 + asz0, 1e-12)
        out['micro_dev_bps'] = (micro - mid) / mid * 1e4
        out['bid_sz0'] = bsz0; out['ask_sz0'] = asz0
        out['bid_n0'] = float(bcn[0]); out['ask_n0'] = float(acn[0])
        nzb = np.flatnonzero(bseg[:4000])[:TOP]
        nza = np.flatnonzero(aseg[:4000])[:TOP]
        tb_px = bb - nzb; ta_px = ba + nza
        tb_sz = bseg[nzb]; ta_sz = aseg[nza]

        def imb(x, y):
            s = x + y
            return (x - y) / s if s > 0 else 0.0
        out['qi1'] = imb(bsz0, asz0)
        out['qi5'] = imb(tb_sz[:5].sum(), ta_sz[:5].sum())
        out['qi10'] = imb(tb_sz.sum(), ta_sz.sum())
        k50 = 0
        for d in BANDS_BPS:
            k = int((mid * d / 1e4 - half) / TICK)
            k = min(max(k, 0), len(cb) - 1, len(ca) - 1)
            out[f'bdep_{d}'] = float(cb[k]); out[f'adep_{d}'] = float(ca[k])
            if d == 10:
                out['bcnt_10'] = float(ccb[k]); out['acnt_10'] = float(cca[k])
            if d == 50:
                out['bcnt_50'] = float(ccb[k]); out['acnt_50'] = float(cca[k])
                k50 = k
        # multi-level OFI (Cont, Cucuringu & Zhang): level-by-rank queue flow vs previous second
        ofi = np.zeros(TOP)
        if self.prev_top is not None:
            pb_px, pb_sz, pa_px, pa_sz = self.prev_top
            m = min(len(tb_px), len(pb_px), len(ta_px), len(pa_px), TOP)
            for i in range(m):
                eb = tb_sz[i] if tb_px[i] > pb_px[i] else (tb_sz[i] - pb_sz[i] if tb_px[i] == pb_px[i] else -pb_sz[i])
                ea = ta_sz[i] if ta_px[i] < pa_px[i] else (ta_sz[i] - pa_sz[i] if ta_px[i] == pa_px[i] else -pa_sz[i])
                ofi[i] = eb - ea
        self.prev_top = (tb_px, tb_sz, ta_px, ta_sz)
        out['ofi1'] = float(ofi[0]); out['ofi5'] = float(ofi[:5].sum()); out['ofi10'] = float(ofi.sum())
        # walls within 50 bps
        jb = int(np.argmax(bseg[:k50 + 1])); ja = int(np.argmax(aseg[:k50 + 1]))
        out['bwall_sz'] = float(bseg[jb]); out['bwall_bps'] = (mid - (bb - jb) * TICK) / mid * 1e4
        out['awall_sz'] = float(aseg[ja]); out['awall_bps'] = ((ba + ja) * TICK - mid) / mid * 1e4
        # sweep impact: bps from mid to the level that completes Q contracts (cap at band edge)
        for q in SWEEP_CONTRACTS:
            jb = min(int(np.searchsorted(cb, q)), len(cb) - 1)
            ja = min(int(np.searchsorted(ca, q)), len(ca) - 1)
            out[f'bimp_{q // 100}'] = (mid - (bb - jb) * TICK) / mid * 1e4
            out[f'aimp_{q // 100}'] = ((ba + ja) * TICK - mid) / mid * 1e4
        return out
