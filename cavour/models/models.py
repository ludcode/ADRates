"""
Multi-curve model for OIS curve construction and scenario analysis.

Provides the main Model class for:
- Building OIS curves from market data (manual or Bloomberg)
- Managing multiple curves simultaneously
- Creating scenario curves with parallel or tenor-specific shocks
- Curve accessor for convenient attribute-style access
"""

from typing import Dict, List
from dataclasses import dataclass, field

from cavour.utils import *
from cavour.trades.rates.ois_curve import OISCurve
from cavour.trades.rates.ois import OIS
from cavour.marketdata.market_data_constants import *
from cavour.marketdata.market_data_engine import MarketCurveBuilder


class CurveAccessor:
    """
    Provides attribute-style access to curves in a Model.

    Allows accessing curves via dot notation (model.curves.GBP_OIS_SONIA)
    instead of dictionary notation (model._curves_dict['GBP_OIS_SONIA']).

    Args:
        curves (Dict[str, OISCurve]): Dictionary of curve name -> OISCurve

    Example:
        >>> model = Model(value_dt)
        >>> model.build_curve("GBP_OIS_SONIA", ...)
        >>> curve = model.curves.GBP_OIS_SONIA  # Attribute access
        >>> curve = model.curves["GBP_OIS_SONIA"]  # Dict access also works
    """
    def __init__(self, curves: Dict[str, OISCurve]):
        self._curves = curves

    def __getattr__(self, item):
        try:
            return self._curves[item]
        except KeyError:
            raise AttributeError(f"No such curve: {item}")

    def __getitem__(self, item):
        return self._curves[item]
    

@dataclass
class Model:
    """
    Multi-curve interest rate model for OIS curve management.

    Central class for building and managing OIS discount curves. Supports:
    - Manual curve construction from rates
    - Automatic Bloomberg data retrieval
    - Scenario analysis with rate shocks
    - Multi-currency FX rate management

    Attributes:
        value_dt (Date): Valuation date for all curves
        _curves_dict (Dict[str, OISCurve]): Internal curve storage
        _curve_params_dict (Dict[str, dict]): Curve construction parameters
        _fx_params_dict (Dict[str, dict]): FX rate data
        _builder (MarketCurveBuilder): Bloomberg data fetcher
        _market_data_used (dict): Historical record of fetched market data

    Example:
        >>> model = Model(Date(30, 4, 2024))
        >>> model.build_curve("GBP_OIS_SONIA", px_list=[5.0, 5.2, 5.5],
        ...                   tenor_list=["1M", "3M", "1Y"])
        >>> swap = OIS(model.value_dt, "10Y", 0.04)
        >>> pos = swap.position(model)
        >>> pv = pos.compute([RequestTypes.VALUE])
    """
    value_dt: Date
    _curves_dict: Dict[str, OISCurve] = field(default_factory=dict)
    _curve_params_dict: Dict[str, dict] = field(default_factory=dict)  # ← Add this line
    _fx_params_dict: Dict[str, dict] = field(default_factory=dict) 
    _builder = MarketCurveBuilder(MARKET_DATA, FX_MARKET_DATA)
    _market_data_used = {}

    def prebuilt_curve(
        self,
        curve_names: Union[str, List[str]],
    ):
        """
        Fetch and build curves from Bloomberg using predefined configurations.

        Args:
            curve_names (Union[str, List[str]]): Curve name(s) from MARKET_DATA
                (e.g., "GBP_OIS_SONIA", ["USD_OIS_SOFR", "EUR_OIS_ESTR"])

        Raises:
            KeyError: If curve_name not found in MARKET_DATA
            Exception: If Bloomberg connection fails

        Example:
            >>> model.prebuilt_curve("GBP_OIS_SONIA")
            >>> # Automatically fetches rates from Bloomberg and builds curve
        """
        #builder = MarketCurveBuilder(MARKET_DATA, FX_MARKET_DATA)

        if isinstance(curve_names, str):
            curve_names = [curve_names]

        for curve_name in curve_names:
            curve_inputs = self._builder.get_curve_inputs(curve_name, self.value_dt)

            # store market data
            self._market_data_used[curve_name] = curve_inputs

            self.build_curve(**curve_inputs)

    def prebuilt_fx(
        self,
        fx_pairs: Union[str, List[str]],
    ):
        """
        Fetch FX rates from Bloomberg.

        Args:
            fx_pairs (Union[str, List[str]]): FX pair(s) (e.g., "EURUSD")
                or ["ALL"] for all configured pairs

        Returns:
            dict: FX rate data with structure {pair: {base, quote, ticker, price}}

        Example:
            >>> fx_data = model.prebuilt_fx(["EURUSD", "GBPUSD"])
        """

        fx_rates = self._builder.get_fx_rates(fx_pairs, self.value_dt)
        self._fx_params_dict.update(fx_rates)     

        return fx_rates


    def build_curve(
        self,
        name: str,
        px_list: List[float],
        tenor_list: List[str],
        spot_days: int = 0,
        swap_type=SwapTypes.PAY,
        fixed_dcc_type=DayCountTypes.ACT_360,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_360,
        bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        interp_type=InterpTypes.LINEAR_ZERO_RATES,
        payment_lag: int = 0,
    ):
        """
        Manually construct an OIS curve from swap rates.

        Args:
            name (str): Curve identifier (e.g., "GBP_OIS_SONIA")
            px_list (List[float]): Swap rates in percentage (e.g., [5.0, 5.2, 5.5])
            tenor_list (List[str]): Tenors (e.g., ["1M", "3M", "1Y"])
            spot_days (int): Settlement lag in business days (default: 0)
            swap_type (SwapTypes): PAY or RECEIVE fixed (default: PAY)
            fixed_dcc_type (DayCountTypes): Fixed leg day count (default: ACT_360)
            fixed_freq_type (FrequencyTypes): Fixed leg frequency (default: ANNUAL)
            float_freq_type (FrequencyTypes): Float leg frequency (default: ANNUAL)
            float_dc_type (DayCountTypes): Float leg day count (default: ACT_360)
            bus_day_type (BusDayAdjustTypes): Business day convention (default: MODIFIED_FOLLOWING)
            interp_type (InterpTypes): Interpolation method (default: LINEAR_ZERO_RATES)
            payment_lag (int): Payment lag in days (default: 0)

        Example:
            >>> model.build_curve(
            ...     name="GBP_OIS_SONIA",
            ...     px_list=[5.19, 5.13, 5.04, 4.75, 4.24],
            ...     tenor_list=["1M", "3M", "6M", "1Y", "5Y"],
            ...     fixed_dcc_type=DayCountTypes.ACT_365F
            ... )
        """
        settle_dt = self.value_dt.add_weekdays(spot_days)
        swaps = [
            OIS(
                effective_dt=settle_dt,
                term_dt_or_tenor=tenor,
                fixed_leg_type=swap_type,
                fixed_coupon=px / 100,
                fixed_freq_type=fixed_freq_type,
                fixed_dc_type=fixed_dcc_type,
                bd_type=bus_day_type,
                float_freq_type=float_freq_type,
                float_dc_type=float_dc_type,
                payment_lag=payment_lag
            )
            for tenor, px in zip(tenor_list, px_list)
        ]

        curve = OISCurve(
            value_dt=self.value_dt,
            ois_swaps=swaps,
            interp_type=interp_type,
            check_refit=True
        )
        self._curves_dict[name] = curve

        # Store parameters for future use (e.g., scenario shock)
        self._curve_params_dict[name] = {
            "tenor_list": tenor_list,
            "px_list": px_list,
            "spot_days": spot_days,
            "swap_type": swap_type,
            "fixed_dcc_type": fixed_dcc_type,
            "fixed_freq_type": fixed_freq_type,
            "float_freq_type": float_freq_type,
            "float_dc_type": float_dc_type,
            "bus_day_type": bus_day_type,
            "interp_type": interp_type,
        }

    def build_fx(self, currency_pairs: list[str], pxs: list[float]) -> dict:
        """
        Manually add FX rates to the model.

        Args:
            currency_pairs (list[str]): FX pair codes (e.g., ["EURUSD", "GBPUSD"])
            pxs (list[float]): Exchange rates corresponding to pairs

        Returns:
            dict: Constructed FX data with base/quote currencies

        Raises:
            ValueError: If invalid currency code in pair

        Example:
            >>> model.build_fx(["EURUSD", "GBPUSD"], [1.08, 1.25])
        """
        result = {}
        for pair, price in zip(currency_pairs, pxs):
            base_code = pair[:3]
            quote_code = pair[3:]

            try:
                base = CurrencyTypes[base_code]
                quote = CurrencyTypes[quote_code]
            except KeyError:
                raise ValueError(f"Invalid currency code in pair: {pair}")

            result[pair] = {
                "base": base,
                "quote": quote,
                "ticker": f"{pair} Curncy",
                "price": float(price)
            }

        self._fx_params_dict.update(result)


    def scenario(self, curve_name: str, shock: dict | float, new_name: str | None = None):
        """
        Create a new model with a shocked curve for scenario analysis.

        Args:
            curve_name (str): Name of curve to shock
            shock (dict | float): Shock specification:
                - float: Parallel shock applied to all tenors (in bps)
                - dict: Tenor-specific shocks (e.g., {"1Y": 10, "5Y": 20})
            new_name (str | None): Name for shocked curve (default: same as curve_name)

        Returns:
            Model: New Model instance with the shocked curve

        Raises:
            ValueError: If curve_name not found in stored parameters

        Example:
            >>> # Parallel 10bp shock
            >>> shocked_model = model.scenario("GBP_OIS_SONIA", 10.0)

            >>> # Tenor-specific shocks
            >>> shocked_model = model.scenario(
            ...     "GBP_OIS_SONIA",
            ...     {"1Y": 5.0, "5Y": 10.0, "10Y": 15.0}
            ... )
        """
        if curve_name not in self._curve_params_dict:
            raise ValueError(f"No stored parameters found for curve '{curve_name}'")

        params = self._curve_params_dict[curve_name]
        base_px = params["px_list"]
        tenors = params["tenor_list"]

        if isinstance(shock, dict):
            shocked_px = [
                base_px[i] + shock.get(tenor, 0.0)
                for i, tenor in enumerate(tenors)
            ]
        else:
            shocked_px = [px + shock for px in base_px]

        # Create new Model with shocked curve
        new_model = Model(value_dt=self.value_dt)
        new_model.build_curve(
            name=new_name or curve_name,
            px_list=shocked_px,
            **{k: v for k, v in params.items() if k not in ("px_list")}
        )

        return new_model

    @property
    def curves(self):
        """
        Access built curves via attribute or dictionary notation.

        Returns:
            CurveAccessor: Accessor providing dot and bracket notation access

        Example:
            >>> model.build_curve("GBP_OIS_SONIA", ...)
            >>> curve1 = model.curves.GBP_OIS_SONIA  # Dot notation
            >>> curve2 = model.curves["GBP_OIS_SONIA"]  # Bracket notation
        """
        return CurveAccessor(self._curves_dict)