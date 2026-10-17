"""Bounded displayed-depth IOC shadow model, never exchange executions.

Hyperliquid public sizes are base-asset sizes. OKX historical contract sizes
cannot enter this model without a separately established unit conversion.
"""
import math


def validate_policy(policy):
    if not isinstance(policy,dict) or set(policy)!={'model','participation_fraction','max_sweep_bps'} or policy['model']!='displayed-depth-ioc-v1':
        raise ValueError('Unsupported shadow liquidity policy')
    for name,lower,upper in (('participation_fraction',0,1),('max_sweep_bps',0,100)):
        value=policy[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lower<value<=upper:
            raise ValueError('Invalid shadow '+name)
    return dict(policy)


class DepthBudget:
    """One observed BTC book's shared consumption ledger for all sleeves."""
    def __init__(self,book,policy):
        self.policy=validate_policy(policy)
        if book.get('coin')!='BTC' or len(book.get('levels',[]))!=2:
            raise ValueError('Depth model requires a BTC two-sided public book')
        self.ts_ms=int(book['time']); self.sides=[]
        for side,levels in enumerate(book['levels']):
            if not 1<=len(levels)<=20:
                raise ValueError('Depth model requires 1..20 observed levels per side')
            converted=[]; previous=None
            for level in levels:
                price=float(level['px']); size=float(level['sz'])
                if not math.isfinite(price) or not math.isfinite(size) or price<=0 or size<=0:
                    raise ValueError('Invalid displayed depth price or size')
                if previous is not None and (price>=previous if side==0 else price<=previous):
                    raise ValueError('Displayed depth must be strictly sorted without duplicate prices')
                converted.append([price,size*self.policy['participation_fraction']]); previous=price
            self.sides.append(converted)
        if self.sides[0][0][0]>=self.sides[1][0][0]:
            raise ValueError('Crossed displayed depth')

    def take(self,side,requested,slippage_bps):
        if side not in (-1,1) or not math.isfinite(requested) or requested<=0 or not math.isfinite(slippage_bps) or slippage_bps<0:
            raise ValueError('Invalid displayed-depth request')
        levels=self.sides[1 if side>0 else 0]
        limit=levels[0][0]*(1+side*self.policy['max_sweep_bps']/10000)
        remaining=requested; notional=0.0; filled=0.0; used=[]
        for level in levels:
            price=level[0]*(1+side*slippage_bps/10000)
            if side*(price-limit)>max(1e-10,limit*1e-12):
                break
            quantity=min(remaining,level[1])
            if quantity<=1e-12:
                continue
            level[1]-=quantity; remaining-=quantity; filled+=quantity; notional+=quantity*price
            used.append({'price':price,'quantity':quantity})
            if remaining<=1e-12:
                break
        return {'requested_quantity':requested,'quantity':filled,'unfilled_quantity':max(0.0,requested-filled),
            'price':notional/filled if filled else None,'levels':used,'model':self.policy['model']}
