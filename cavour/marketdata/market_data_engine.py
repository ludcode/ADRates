import os

# When Bloomberg is disabled (e.g., on Mac), stub 'blpapi' so xbbg import won't explode
if os.getenv("BLOOMBERG_DISABLED") == "1":
    import sys, types
    sys.modules.setdefault("blpapi", types.ModuleType("blpapi"))

try:
    stop #from xbbg import blp  # type: ignore
except Exception:
    class _BLPStub:
        def __getattr__(self, name):
            raise ImportError("xbbg/blpapi not available on this machine")
    blp = _BLPStub()

from typing import Dict, Optional, Tuple, List
from cavour.utils.date import Date
from cavour.utils.global_types import SwapTypes
import heapq
import math




class MarketCurveBuilder:
    def __init__(self, market_data: Dict[str, dict],
                 fx_market_data: Dict[str, dict]):
        self.market_data = market_data
        self.fx_market_data = fx_market_data

    def get_curve_inputs(self, curve_key: str, value_date: Date):
        value_dt = value_date.datetime()
        curve_def = self.market_data[curve_key]
        tickers_dict = curve_def["tickers"]
        conventions = curve_def["conventions"]

        tenor_list = list(tickers_dict.keys())
        ticker_list = list(tickers_dict.values())

        field = "PX_LAST"

        # Fetch prices
        df = blp.bdh(
            tickers=ticker_list,
            flds=field,
            start_date=value_dt,
            end_date=value_dt,
            Per="D"
        )

        # Extract final px_list
        px_list = [df[ticker][field].iloc[0] for ticker in ticker_list]

        return {
            "name": curve_key,
            "px_list": px_list,
            "tenor_list": tenor_list,
            "spot_days": 0,
            "swap_type": SwapTypes.PAY,
            "fixed_dcc_type": conventions["fixed_day_count"],
            "fixed_freq_type": conventions["fixed_frequency"],
            "float_freq_type": conventions["float_frequency"],
            "float_dc_type": conventions["float_day_count"],
            "bus_day_type": conventions["business_day_adjustment"],
            "interp_type": conventions["interp_type"]
        }
    
    def get_fx_rates(self, fx_key: list[str], value_date: Date):
        value_dt = value_date.datetime()
        field = "PX_LAST"

        if fx_key == ["ALL"]:
            fx_return = self.fx_market_data
            ticker_list = []
            tickers = []
            for pair, details in self.fx_market_data.items():
                ticker_list.append(details['ticker'])
                tickers.append(pair)
        else:
            fx_return = {key:val for key,val in self.fx_market_data.items() if key in fx_key}
            fx_ticker_dict = {key:val['ticker'] for key,val in self.fx_market_data.items() if key in fx_key}
            ticker_list = list(fx_ticker_dict.values())
            tickers = list(fx_ticker_dict.keys())

        # Fetch prices
        df = blp.bdh(
            tickers=ticker_list,
            flds=field,
            start_date=value_dt,
            end_date=value_dt,
            Per="D"
        )

        px_list = [df[ticker][field].iloc[0] for ticker in ticker_list]

        fx_pairs = dict(zip(tickers,px_list))

        for pair, price in fx_pairs.items():
            if pair in fx_return:
                fx_return[pair]["price"] = float(price)


        return fx_return

    

class FXRoutingEngine:
    def __init__(self):
        self._fx_rates: Dict[str, float] = {}         # e.g., EURUSD = 1.08
        self._graph: Dict[str, Dict[str, float]] = {} # adjacency list: EUR -> {USD: 1.08}
        self._overrides: Dict[str, str] = {}          # manual override: PLN -> EUR

    def set_fx_rate(self, pair: str, rate: float):
        pair = pair.upper()
        ccy1, ccy2 = pair[:3], pair[3:]
        self._fx_rates[pair] = rate

        # Update graph
        self._graph.setdefault(ccy1, {})[ccy2] = rate
        self._graph.setdefault(ccy2, {})[ccy1] = 1.0 / rate

    def set_bulk_fx_rates(self, fx_dict: Dict[str, float]):
        for k, v in fx_dict.items():
            self.set_fx_rate(k, v)

    def set_override(self, ccy: str, via: str):
        self._overrides[ccy.upper()] = via.upper()

    def _dijkstra(self, src: str, tgt: str) -> Tuple[Optional[float], List[str]]:
        src, tgt = src.upper(), tgt.upper()
        if src not in self._graph or tgt not in self._graph:
            return None, []

        # log conversion to turn multiplication into addition
        visited = set()
        heap = [(0, src, [])]  # log(price), current_ccy, path

        while heap:
            log_cost, current, path = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            path = path + [current]

            if current == tgt:
                return math.exp(-log_cost), path

            for neighbor, rate in self._graph.get(current, {}).items():
                if neighbor not in visited:
                    heapq.heappush(heap, (log_cost - math.log(rate), neighbor, path))

        return None, []

    def get_cross_rate(self, from_ccy: str, to_ccy: str) -> Optional[float]:
        from_ccy, to_ccy = from_ccy.upper(), to_ccy.upper()

        # Check if override forces from_ccy to go via intermediate
        via = self._overrides.get(from_ccy)
        if via and via != to_ccy:
            # First hop: from_ccy → via
            r1, _ = self._dijkstra(from_ccy, via)
            # Second hop: via → to_ccy
            r2, _ = self._dijkstra(via, to_ccy)
            if r1 and r2:
                return r1 * r2
            return None

        # No override or direct to override
        return self._dijkstra(from_ccy, to_ccy)[0]

    def get_cross_rate_with_path(self, from_ccy: str, to_ccy: str) -> Tuple[Optional[float], List[str]]:
        from_ccy, to_ccy = from_ccy.upper(), to_ccy.upper()

        via = self._overrides.get(from_ccy)

        # If an override exists and it's not the final destination, enforce two-step routing
        if via and via != to_ccy:
            r1, path1 = self._dijkstra(from_ccy, via)
            r2, path2 = self._dijkstra(via, to_ccy)
            if r1 and r2:
                full_path = path1 + path2[1:]  # Avoid repeating 'via'
                return r1 * r2, full_path
            return None, []

        # No override or override points directly to destination
        return self._dijkstra(from_ccy, to_ccy)
    