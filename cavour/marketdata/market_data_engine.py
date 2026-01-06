"""
Market data engine for fetching Bloomberg data and computing FX cross rates.

Provides two main classes:
- MarketCurveBuilder: Fetches OIS curve and FX data from Bloomberg
- FXRoutingEngine: Computes cross FX rates using graph-based routing
"""

from typing import Dict, Optional, Tuple, List
from xbbg import blp
from cavour.utils import *
import heapq
import math


class MarketCurveBuilder:
    """
    Fetches market data from Bloomberg for curve construction.

    Uses xbbg library to retrieve historical prices for OIS swaps and FX rates
    based on predefined ticker mappings and market conventions.

    Attributes:
        market_data (Dict[str, dict]): OIS curve definitions with tickers and conventions
        fx_market_data (Dict[str, dict]): FX pair definitions with tickers
    """
    def __init__(self, market_data: Dict[str, dict],
                 fx_market_data: Dict[str, dict]):
        """
        Initialize the market data builder.

        Args:
            market_data (Dict[str, dict]): Curve definitions from MARKET_DATA constant
            fx_market_data (Dict[str, dict]): FX definitions from FX_MARKET_DATA constant
        """
        self.market_data = market_data
        self.fx_market_data = fx_market_data

    def get_curve_inputs(self, curve_key: str, value_date: Date):
        """
        Fetch curve construction inputs from Bloomberg.

        Retrieves swap rates for all tenors defined in the curve configuration
        and packages them with market conventions for curve building.

        Args:
            curve_key (str): Curve name (e.g., "GBP_OIS_SONIA", "USD_OIS_SOFR")
            value_date (Date): Valuation date for market data retrieval

        Returns:
            dict: Curve construction parameters including:
                - name: Curve identifier
                - px_list: List of swap rates retrieved from Bloomberg
                - tenor_list: List of tenors (e.g., ["1M", "3M", "1Y"])
                - spot_days: Settlement lag
                - swap_type: Pay or receive fixed
                - Market conventions (day count, frequency, etc.)

        Raises:
            KeyError: If curve_key not found in market_data
            Exception: If Bloomberg connection fails or data unavailable
        """
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
        """
        Fetch FX spot rates from Bloomberg.

        Args:
            fx_key (list[str]): List of FX pairs (e.g., ["EURUSD", "GBPUSD"])
                               or ["ALL"] to fetch all available pairs
            value_date (Date): Valuation date for FX rate retrieval

        Returns:
            dict: FX rate data with structure:
                {pair: {base, quote, ticker, price}}

        Example:
            >>> builder.get_fx_rates(["EURUSD"], Date(15, 6, 2023))
            {'EURUSD': {'base': CurrencyTypes.EUR, 'quote': CurrencyTypes.USD,
                        'ticker': 'EURUSD Curncy', 'price': 1.0856}}
        """
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

    def get_xccy_curve_inputs(self, xccy_curve_key: str, value_date: Date):
        """
        Fetch XCCY curve construction inputs from Bloomberg.

        Retrieves basis spreads and all required components for XCCY curve building:
        - Domestic and foreign OIS curve data
        - XCCY basis spreads
        - FX spot rate

        Args:
            xccy_curve_key (str): XCCY curve name (e.g., "GBPUSD_XCCY_SONIA_SOFR")
            value_date (Date): Valuation date for market data retrieval

        Returns:
            dict: XCCY curve construction parameters including:
                - name: XCCY curve identifier
                - domestic_curve_name: Domestic OIS curve name (e.g., "USD_OIS_SOFR")
                - foreign_curve_name: Foreign OIS curve name (e.g., "GBP_OIS_SONIA")
                - domestic_curve_inputs: Dict with domestic OIS curve data
                - foreign_curve_inputs: Dict with foreign OIS curve data
                - basis_spreads: List of basis spreads (in bps)
                - tenor_list: List of tenors for basis swaps
                - spot_fx: FX spot rate
                - fx_pair: FX pair name (e.g., "GBPUSD")
                - domestic_freq_type: Domestic leg payment frequency
                - foreign_freq_type: Foreign leg payment frequency
                - domestic_dc_type: Domestic leg day count
                - foreign_dc_type: Foreign leg day count
                - bus_day_type: Business day adjustment
                - interp_type: Interpolation method

        Raises:
            KeyError: If xccy_curve_key not found in market_data
            ValueError: If curve name format is invalid
            Exception: If Bloomberg connection fails or data unavailable

        Example:
            >>> inputs = builder.get_xccy_curve_inputs("GBPUSD_XCCY_SONIA_SOFR", value_dt)
            >>> # Returns all data needed to build GBP/USD XCCY curve
        """
        # Get XCCY curve definition
        if xccy_curve_key not in self.market_data:
            raise KeyError(f"XCCY curve '{xccy_curve_key}' not found in market data")

        xccy_def = self.market_data[xccy_curve_key]

        # Verify this is an XCCY curve
        if xccy_def.get("type") != "XCCY":
            raise ValueError(f"Curve '{xccy_curve_key}' is not an XCCY curve (type: {xccy_def.get('type')})")

        # Parse curve name: "GBPUSD_XCCY_SONIA_SOFR"
        # Format: {FOREIGN}{DOMESTIC}_XCCY_{FOREIGN_INDEX}_{DOMESTIC_INDEX}
        parts = xccy_curve_key.split("_")
        if len(parts) < 4:
            raise ValueError(f"Invalid XCCY curve name format: {xccy_curve_key}")

        # Extract currency pair (e.g., "GBPUSD")
        fx_pair = parts[0]
        if len(fx_pair) != 6:
            raise ValueError(f"Invalid FX pair in curve name: {fx_pair}")

        foreign_ccy = fx_pair[:3]  # GBP
        domestic_ccy = fx_pair[3:]  # USD

        # Extract indices
        foreign_index = parts[2]  # SONIA
        domestic_index = parts[3]  # SOFR

        # Construct OIS curve names
        foreign_curve_name = f"{foreign_ccy}_OIS_{foreign_index}"
        domestic_curve_name = f"{domestic_ccy}_OIS_{domestic_index}"

        # Fetch XCCY basis spreads
        value_dt = value_date.datetime()
        tickers_dict = xccy_def["tickers"]
        tenor_list = list(tickers_dict.keys())
        ticker_list = list(tickers_dict.values())

        field = "PX_LAST"
        df = blp.bdh(
            tickers=ticker_list,
            flds=field,
            start_date=value_dt,
            end_date=value_dt,
            Per="D"
        )

        basis_spreads = [df[ticker][field].iloc[0] for ticker in ticker_list]

        # Fetch domestic OIS curve
        domestic_curve_inputs = self.get_curve_inputs(domestic_curve_name, value_date)

        # Fetch foreign OIS curve
        foreign_curve_inputs = self.get_curve_inputs(foreign_curve_name, value_date)

        # Fetch FX spot rate
        fx_data = self.get_fx_rates([fx_pair], value_date)
        spot_fx = fx_data[fx_pair]["price"]

        # Get conventions
        conventions = xccy_def["conventions"]

        # For XCCY, foreign leg typically uses ACT_365F for GBP
        # Domestic leg uses conventions from the config
        if foreign_ccy == "GBP":
            foreign_dc_type = DayCountTypes.ACT_365F
        else:
            foreign_dc_type = conventions.get("float_day_count", DayCountTypes.ACT_360)

        return {
            "name": xccy_curve_key,
            "domestic_curve_name": domestic_curve_name,
            "foreign_curve_name": foreign_curve_name,
            "domestic_curve_inputs": domestic_curve_inputs,
            "foreign_curve_inputs": foreign_curve_inputs,
            "basis_spreads": basis_spreads,
            "tenor_list": tenor_list,
            "spot_fx": spot_fx,
            "fx_pair": fx_pair,
            "domestic_freq_type": conventions["fixed_frequency"],
            "foreign_freq_type": conventions["float_frequency"],
            "domestic_dc_type": conventions["fixed_day_count"],
            "foreign_dc_type": foreign_dc_type,
            "bus_day_type": conventions["business_day_adjustment"],
            "interp_type": conventions["interp_type"],
        }



class FXRoutingEngine:
    """
    Computes cross FX rates using graph-based shortest path routing.

    Uses Dijkstra's algorithm to find optimal conversion paths through
    available FX pairs. Supports manual routing overrides for specific
    currencies (e.g., force PLN to convert via EUR).

    Attributes:
        _fx_rates (Dict[str, float]): Direct FX rates (e.g., EURUSD = 1.08)
        _graph (Dict[str, Dict[str, float]]): Adjacency list for graph traversal
        _overrides (Dict[str, str]): Manual routing rules (e.g., PLN -> EUR)

    Example:
        >>> engine = FXRoutingEngine()
        >>> engine.set_fx_rate("EURUSD", 1.08)
        >>> engine.set_fx_rate("GBPUSD", 1.25)
        >>> rate = engine.get_cross_rate("GBP", "EUR")  # Returns 1.157
    """
    def __init__(self):
        self._fx_rates: Dict[str, float] = {}         # e.g., EURUSD = 1.08
        self._graph: Dict[str, Dict[str, float]] = {} # adjacency list: EUR -> {USD: 1.08}
        self._overrides: Dict[str, str] = {}          # manual override: PLN -> EUR

    def set_fx_rate(self, pair: str, rate: float):
        """
        Add an FX rate to the routing graph.

        Args:
            pair (str): Currency pair (e.g., "EURUSD")
            rate (float): Exchange rate (e.g., 1.08 means 1 EUR = 1.08 USD)

        Note:
            Automatically creates bidirectional edges in the graph
            (e.g., EUR->USD and USD->EUR with inverted rate).
        """
        pair = pair.upper()
        ccy1, ccy2 = pair[:3], pair[3:]
        self._fx_rates[pair] = rate

        # Update graph
        self._graph.setdefault(ccy1, {})[ccy2] = rate
        self._graph.setdefault(ccy2, {})[ccy1] = 1.0 / rate

    def set_bulk_fx_rates(self, fx_dict: Dict[str, float]):
        """
        Add multiple FX rates at once.

        Args:
            fx_dict (Dict[str, float]): Dictionary of {pair: rate}
        """
        for k, v in fx_dict.items():
            self.set_fx_rate(k, v)

    def set_override(self, ccy: str, via: str):
        """
        Force a currency to route through an intermediate currency.

        Args:
            ccy (str): Currency to override (e.g., "PLN")
            via (str): Intermediate currency (e.g., "EUR")

        Example:
            >>> engine.set_override("PLN", "EUR")
            >>> # PLN->USD will now route as PLN->EUR->USD
        """
        self._overrides[ccy.upper()] = via.upper()

    def _dijkstra(self, src: str, tgt: str) -> Tuple[Optional[float], List[str]]:
        """
        Find shortest path and exchange rate between two currencies.

        Uses Dijkstra's algorithm in log space to find the optimal conversion path.
        Log transformation converts multiplication (chaining FX rates) into addition.

        Args:
            src (str): Source currency code (e.g., "EUR")
            tgt (str): Target currency code (e.g., "USD")

        Returns:
            tuple: (exchange_rate, path)
                - exchange_rate: Float or None if no path exists
                - path: List of currency codes in conversion path

        Note:
            Returns (None, []) if either currency not in graph.
        """
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
        """
        Get the cross exchange rate between two currencies.

        Args:
            from_ccy (str): Source currency code
            to_ccy (str): Target currency code

        Returns:
            Optional[float]: Exchange rate or None if no path exists

        Example:
            >>> engine.get_cross_rate("GBP", "EUR")
            1.157
        """
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
        """
        Get cross exchange rate with the conversion path.

        Args:
            from_ccy (str): Source currency code
            to_ccy (str): Target currency code

        Returns:
            tuple: (exchange_rate, path)
                - exchange_rate: Float or None if no path exists
                - path: List of currency codes showing conversion route

        Example:
            >>> engine.get_cross_rate_with_path("GBP", "JPY")
            (188.5, ['GBP', 'USD', 'JPY'])
        """
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
    