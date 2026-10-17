"""Identical event driven broker for historical replay and live shadow execution.

L2 depth reductions are NEVER called traded volume. Passive fills use a strict
trade-through proxy and remain research-only without trade/queue evidence.
"""
from dataclasses import dataclass
import math
import numpy as np
import polars as pl

@dataclass
class Account:
    cash: float
    quantity: float=0.0
    entry_price: float=0.0
    entry_time: int=0
    entry_fee: float=0.0
    entry_mid: float=0.0
    realized: float=0.0
    fees: float=0.0
    halted: bool=False
    entry_quantity: float=0.0
    exit_quantity: float=0.0
    exit_notional: float=0.0
    exit_mid_notional: float=0.0
    exit_fees: float=0.0

class Broker:
    def __init__(self,spec,execution,liquidity_policy=None):
        self.spec=spec
        self.config=execution
        from .liquidity import validate_policy
        self.liquidity_policy=validate_policy(liquidity_policy) if liquidity_policy is not None else None
        self.account=Account(execution['initial_equity'])
        self.initial=execution['initial_equity']
        self.peak=self.initial
        self.pending=None
        self.last_ts=None
        self.trades=[]
        self.events=[]
        self.equity=[]
        self.orders=0
        self.funding_boundary_crossings=0
        self.closing=False

    def mark(self,book):
        a=self.account
        return a.cash+a.quantity*book['mid_price']

    def fill(self,book,side,quantity,maker=False,price=None):
        a=self.account
        if not math.isfinite(quantity) or quantity<=0 or side not in (-1,1):
            raise ValueError('Invalid fill size or side')
        if abs(a.quantity)>1e-12 and (side*a.quantity>=0 or quantity>abs(a.quantity)+1e-10):
            raise ValueError('Fill must reduce the actual open position')
        details={}
        if self.liquidity_policy is not None:
            if maker:
                raise ValueError('Displayed depth cannot certify a passive fill')
            depth=book.get('_liquidity')
            if depth is None or depth.ts_ms!=book['ts_ms'] or depth.policy!=self.liquidity_policy:
                raise ValueError('Missing matching frozen displayed-depth budget')
            details=depth.take(side,quantity,self.config['slippage_bps'])
            if details['quantity']<=1e-12:
                self.events.append({'kind':'IOC_NO_FILL','ts_ms':book['ts_ms'],'side':side,**details})
                return 0.0
            quantity=details['quantity']; price=details['price']
        reference=price if price is not None else book['ask_p0'] if side>0 else book['bid_p0']
        price=reference if maker or self.liquidity_policy is not None else reference*(1+side*self.config['slippage_bps']/1e4)
        fee=quantity*price*(self.config['maker_fee_bps'] if maker else self.config['taker_fee_bps'])/1e4
        signed=side*quantity
        a.cash-=signed*price+fee
        a.fees+=fee
        if abs(a.quantity)<1e-12:
            a.quantity=signed; a.entry_price=price; a.entry_time=book['ts_ms']; a.entry_fee=fee; a.entry_mid=book['mid_price']
            a.entry_quantity=quantity; a.exit_quantity=0.0; a.exit_notional=0.0; a.exit_mid_notional=0.0; a.exit_fees=0.0
        else:
            direction=1 if a.quantity>0 else -1
            a.realized+=direction*quantity*(price-a.entry_price)-a.entry_fee*quantity/a.entry_quantity-fee
            a.exit_quantity+=quantity; a.exit_notional+=quantity*price; a.exit_mid_notional+=quantity*book['mid_price']; a.exit_fees+=fee
            a.quantity+=signed
            self.closing=abs(a.quantity)>=1e-12
            if abs(a.quantity)<1e-12:
                exit_price=a.exit_notional/a.exit_quantity; exit_mid=a.exit_mid_notional/a.exit_quantity
                gross=direction*a.entry_quantity*(exit_price-a.entry_price)
                pnl=gross-a.entry_fee-a.exit_fees
                gross_mid_pnl=direction*a.entry_quantity*(exit_mid-a.entry_mid)
                entry_notional=a.entry_quantity*a.entry_mid
                self.trades.append({'entry_ts':a.entry_time,'exit_ts':book['ts_ms'],'side':direction,'quantity':a.entry_quantity,'entry_price':a.entry_price,'exit_price':exit_price,'entry_mid':a.entry_mid,'exit_mid':exit_mid,'entry_notional':entry_notional,'gross_mid_pnl':gross_mid_pnl,'gross_mid_bps':gross_mid_pnl/entry_notional*1e4,'execution_drag':gross_mid_pnl-gross,'net_bps':pnl/entry_notional*1e4,'pnl':pnl,'fees':a.entry_fee+a.exit_fees})
                a.quantity=0.0; a.entry_fee=0.0
        self.events.append({'kind':'FILL','ts_ms':book['ts_ms'],'side':side,'quantity':quantity,'price':price,'fee':fee,'maker_proxy':maker,**details})
        return quantity

    def step(self,book,signal,terminal=False,decision_ts_ms=None):
        if not math.isfinite(signal) or abs(signal)>1.000001:
            raise ValueError('Invalid execution signal')
        ts=int(book['ts_ms'])
        if decision_ts_ms is not None and not math.isfinite(decision_ts_ms):
            raise ValueError('Invalid signal decision timestamp')
        decision_ts=max(ts,int(decision_ts_ms)) if decision_ts_ms is not None else ts
        if self.last_ts is not None and ts<self.last_ts:
            raise ValueError('Out of order broker event')
        gap=self.last_ts is not None and ts-self.last_ts>self.config['max_gap_ms']
        duplicate=self.last_ts is not None and ts==self.last_ts
        if self.last_ts is not None and self.last_ts//3600000!=ts//3600000 and abs(self.account.quantity)>1e-12:
            self.funding_boundary_crossings+=1
            self.events.append({'kind':'UNMODELED_FUNDING_BOUNDARY','ts_ms':ts})
        self.last_ts=ts
        a=self.account
        spread_ok=(book['ask_p0']>book['bid_p0']>0 and book['spread_bps']<=self.config['max_spread_bps'])
        equity=self.mark(book)
        self.peak=max(self.peak,equity)
        drawdown=(self.peak-equity)/self.peak
        if drawdown>=self.config['max_drawdown_fraction']:
            a.halted=True
        # Gaps cancel orders; exposure is marked/closed at the first fresh quote.
        if terminal or gap or a.halted:
            self.pending=None
            if abs(a.quantity)>1e-12:
                self.fill(book,-1 if a.quantity>0 else 1,abs(a.quantity))
            if gap:
                self.events.append({'kind':'DATA_GAP','ts_ms':ts})
        elif not duplicate:
            if self.pending:
                order=self.pending
                if order['kind']=='entry' and not spread_ok:
                    self.pending=None
                elif ts>=order['arrival']:
                    if order['kind']=='exit':
                        self.fill(book,order['side'],abs(a.quantity)); self.pending=None
                    elif self.spec['mode']=='taker':
                        qty=self.initial*self.config['allocation_fraction']/book['mid_price']
                        self.fill(book,order['side'],qty); self.pending=None
                    else:
                        # Snapshot data cannot prove queue depletion. Only strict price trade-through.
                        through=(order['side']>0 and book['ask_p0']<order['price']) or (order['side']<0 and book['bid_p0']>order['price'])
                        if through:
                            qty=self.initial*self.config['allocation_fraction']/order['price']
                            self.fill(book,order['side'],qty,True,order['price']); self.pending=None
                        elif ts-order['arrival']>=15000:
                            self.pending=None
            target=1 if signal>self.spec['threshold'] else -1 if signal < -self.spec['threshold'] else 0
            if self.pending is None and abs(a.quantity)>1e-12:
                expired=ts-a.entry_time>=self.spec['holding_seconds']*1000
                flip=target!=0 and target*a.quantity<0
                if expired or flip or not spread_ok or self.closing:
                    self.pending={'kind':'exit','side':-1 if a.quantity>0 else 1,'arrival':decision_ts+self.config['latency_ms'],'decision_ts':decision_ts}
                    self.orders+=1
            elif self.pending is None and not a.halted and spread_ok and target and (
                ts//3600000==decision_ts//3600000 and
                3600000-decision_ts%3600000>self.spec['holding_seconds']*1000+2*self.config['latency_ms']+self.config['max_gap_ms']):
                # Historical funding is absent. Do not deliberately establish
                # exposure that would straddle the target venue's hourly funding.
                self.pending={'kind':'entry','side':target,'price':book['bid_p0'] if target>0 else book['ask_p0'],'arrival':decision_ts+self.config['latency_ms'],'decision_ts':decision_ts}
                self.orders+=1
        marked=self.mark(book)
        self.peak=max(self.peak,marked)
        if (self.peak-marked)/self.peak>=self.config['max_drawdown_fraction']:
            a.halted=True
        self.equity.append({'ts_ms':ts,'equity':marked,'quantity':a.quantity})
        return marked

    def summary(self):
        equity=np.array([r['equity'] for r in self.equity] or [self.initial])
        peaks=np.maximum.accumulate(np.r_[self.initial,equity])[1:]
        pnl=np.array([t['pnl'] for t in self.trades])
        return {'trades':len(self.trades),'orders':self.orders,'return_fraction':float(equity[-1]/self.initial-1),'pnl':float(equity[-1]-self.initial),'fees':self.account.fees,'max_drawdown_fraction':float(np.max((peaks-equity)/peaks)),'win_rate':float((pnl>0).mean()) if len(pnl) else 0.0,'halted':self.account.halted,'fill_model':self.liquidity_policy['model'] if self.liquidity_policy else 'taker_cross_spread' if self.spec['mode']=='taker' else 'unverified_passive_trade_through_proxy','funding_included':False,'funding_boundary_crossings':self.funding_boundary_crossings}

def replay(frame,signals,spec,execution):
    if len(frame)!=len(signals) or len(frame)<2:
        raise ValueError('Replay needs matching signals and at least two book events')
    broker=Broker(spec,execution)
    previous_segment=None
    warmup=spec.get('warmup_rows',300)
    segment_count=0
    for i,book in enumerate(frame.iter_rows(named=True)):
        if previous_segment!=book.get('segment',0):
            # Never carry fills or exposure through reconstruction resets.
            if previous_segment is not None:
                broker.step(book,0.0,terminal=True)
            previous_segment=book.get('segment',0); segment_count=0
        segment_count+=1
        signal=float(signals[i]) if segment_count>warmup else 0.0
        broker.step(book,signal,terminal=i==len(frame)-1)
    equity=pl.DataFrame(broker.equity).unique(subset=['ts_ms'],keep='last',maintain_order=True)
    equity=equity.with_columns((pl.col('equity').pct_change().fill_null(0.0)).alias('return'))
    return broker.summary(),equity,broker.trades
