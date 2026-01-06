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
from cavour.trades.rates.xccy_curve import XccyCurve
from cavour.trades.rates.xccy_basis_swap import XccyBasisSwap
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

        # Extract curve type and currency from the curve name
        curve_type = CurveTypes[name]
        # Extract currency from the first 3 characters of the curve name (e.g., "GBP" from "GBP_OIS_SONIA")
        currency_code = name.split('_')[0]
        currency = CurrencyTypes[currency_code]

        swaps = [
            OIS(
                effective_dt=settle_dt,
                term_dt_or_tenor=tenor,
                fixed_leg_type=swap_type,
                fixed_coupon=px / 100,
                fixed_freq_type=fixed_freq_type,
                fixed_dc_type=fixed_dcc_type,
                floating_index=curve_type,
                currency=currency,
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

    def build_xccy_curve(
        self,
        name: str,
        domestic_curve_name: str,
        foreign_curve_name: str,
        basis_spreads: List[float],
        tenor_list: List[str],
        spot_fx: float,
        domestic_notional: float = 100_000_000,
        domestic_freq_type: FrequencyTypes = FrequencyTypes.ANNUAL,
        foreign_freq_type: FrequencyTypes = FrequencyTypes.ANNUAL,
        domestic_dc_type: DayCountTypes = DayCountTypes.ACT_360,
        foreign_dc_type: DayCountTypes = DayCountTypes.ACT_365F,
        bus_day_type: BusDayAdjustTypes = BusDayAdjustTypes.MODIFIED_FOLLOWING,
        interp_type: InterpTypes = InterpTypes.FLAT_FWD_RATES,
        use_ad: bool = True,
    ):
        """
        Build a cross-currency basis swap curve from basis spreads.

        Args:
            name (str): XCCY curve identifier (e.g., "GBP_USD_BASIS")
            domestic_curve_name (str): Domestic OIS curve name (e.g., "USD_OIS_SOFR")
            foreign_curve_name (str): Foreign OIS curve name (e.g., "GBP_OIS_SONIA")
            basis_spreads (List[float]): Basis spreads in bps (e.g., [-0.88, -11.62])
            tenor_list (List[str]): Tenors (e.g., ["10Y", "20Y"])
            spot_fx: FX spot rate (foreign/domestic, e.g., GBPUSD = 1.3468)
            domestic_notional (float): Notional for calibration swaps (default: 100M)
            domestic_freq_type (FrequencyTypes): Domestic leg frequency
            foreign_freq_type (FrequencyTypes): Foreign leg frequency
            domestic_dc_type (DayCountTypes): Domestic leg day count
            foreign_dc_type (DayCountTypes): Foreign leg day count
            bus_day_type (BusDayAdjustTypes): Business day convention
            interp_type (InterpTypes): Interpolation method
            use_ad (bool): Enable JAX automatic differentiation (default: True)

        Raises:
            ValueError: If domestic or foreign curve not found in model

        Example:
            >>> model.build_xccy_curve(
            ...     name="GBP_USD_BASIS",
            ...     domestic_curve_name="USD_OIS_SOFR",
            ...     foreign_curve_name="GBP_OIS_SONIA",
            ...     basis_spreads=[-0.88, -11.62],
            ...     tenor_list=["10Y", "20Y"],
            ...     spot_fx=1.3468
            ... )
        """
        # Get domestic and foreign curves from model
        if domestic_curve_name not in self._curves_dict:
            raise ValueError(f"Domestic curve '{domestic_curve_name}' not found in model. "
                           f"Build it first using build_curve() or prebuilt_curve().")

        if foreign_curve_name not in self._curves_dict:
            raise ValueError(f"Foreign curve '{foreign_curve_name}' not found in model. "
                           f"Build it first using build_curve() or prebuilt_curve().")

        domestic_curve = self._curves_dict[domestic_curve_name]
        foreign_curve = self._curves_dict[foreign_curve_name]

        # Extract currency and index information from curve names
        # E.g., "USD_OIS_SOFR" -> currency=USD, index=USD_OIS_SOFR
        domestic_currency_code = domestic_curve_name.split('_')[0]
        foreign_currency_code = foreign_curve_name.split('_')[0]

        domestic_currency = CurrencyTypes[domestic_currency_code]
        foreign_currency = CurrencyTypes[foreign_currency_code]

        domestic_index = CurveTypes[domestic_curve_name]
        foreign_index = CurveTypes[foreign_curve_name]

        # Calculate foreign notional from domestic notional and spot FX
        foreign_notional = domestic_notional / spot_fx

        # Create XccyBasisSwap calibration instruments
        basis_swaps = []
        for tenor, spread_bps in zip(tenor_list, basis_spreads):
            swap = XccyBasisSwap(
                effective_dt=self.value_dt,
                term_dt_or_tenor=tenor,
                domestic_notional=domestic_notional,
                foreign_notional=foreign_notional,
                domestic_spread=0.0,
                foreign_spread=spread_bps / 10000.0,  # Convert bps to decimal
                domestic_freq_type=domestic_freq_type,
                foreign_freq_type=foreign_freq_type,
                domestic_dc_type=domestic_dc_type,
                foreign_dc_type=foreign_dc_type,
                domestic_floating_index=domestic_index,
                foreign_floating_index=foreign_index,
                domestic_currency=domestic_currency,
                foreign_currency=foreign_currency
            )
            basis_swaps.append(swap)

        # Build XCCY curve
        xccy_curve = XccyCurve(
            value_dt=self.value_dt,
            basis_swaps=basis_swaps,
            domestic_curve=domestic_curve,
            foreign_curve=foreign_curve,
            spot_fx=1/spot_fx,  # XccyCurve expects USD/GBP (inverse)
            interp_type=interp_type,
            use_ad=use_ad
        )

        self._curves_dict[name] = xccy_curve

        # Store parameters for scenario analysis
        self._curve_params_dict[name] = {
            "domestic_curve_name": domestic_curve_name,
            "foreign_curve_name": foreign_curve_name,
            "basis_spreads": basis_spreads,
            "tenor_list": tenor_list,
            "spot_fx": spot_fx,
            "domestic_notional": domestic_notional,
            "domestic_freq_type": domestic_freq_type,
            "foreign_freq_type": foreign_freq_type,
            "domestic_dc_type": domestic_dc_type,
            "foreign_dc_type": foreign_dc_type,
            "bus_day_type": bus_day_type,
            "interp_type": interp_type,
            "use_ad": use_ad,
        }

    def prebuilt_xccy_curve(self, curve_name: str):
        """
        Fetch and build XCCY curve from Bloomberg using predefined configurations.

        Automatically fetches basis spreads, builds required OIS curves,
        and constructs the XCCY curve using the same approach as manual construction.

        Args:
            curve_name (str): XCCY curve name from MARKET_DATA
                (e.g., "GBPUSD_XCCY_SONIA_SOFR")

        Raises:
            KeyError: If curve_name not found in MARKET_DATA
            ValueError: If curve is not an XCCY type
            Exception: If Bloomberg connection fails

        Example:
            >>> model.prebuilt_xccy_curve("GBPUSD_XCCY_SONIA_SOFR")
            >>> # Automatically builds GBP_OIS_SONIA, USD_OIS_SOFR, and XCCY curve
        """
        # Fetch XCCY curve inputs from Bloomberg
        xccy_inputs = self._builder.get_xccy_curve_inputs(curve_name, self.value_dt)

        # Store market data
        self._market_data_used[curve_name] = xccy_inputs

        domestic_curve_name = xccy_inputs["domestic_curve_name"]
        foreign_curve_name = xccy_inputs["foreign_curve_name"]

        # Build domestic OIS curve in SEPARATE model (matching manual approach)
        # IMPORTANT: Override interp_type to FLAT_FWD_RATES for consistency with manual construction
        # Market data defaults to LINEAR_ZERO_RATES but XCCY curves require FLAT_FWD_RATES OIS inputs
        domestic_model = Model(self.value_dt)
        domestic_inputs = xccy_inputs["domestic_curve_inputs"].copy()
        domestic_inputs["interp_type"] = InterpTypes.FLAT_FWD_RATES
        domestic_model.build_curve(**domestic_inputs)

        # Build foreign OIS curve in SEPARATE model (matching manual approach)
        foreign_model = Model(self.value_dt)
        foreign_inputs = xccy_inputs["foreign_curve_inputs"].copy()
        foreign_inputs["interp_type"] = InterpTypes.FLAT_FWD_RATES
        foreign_model.build_curve(**foreign_inputs)

        # Extract curves from separate models
        domestic_curve = domestic_model._curves_dict[domestic_curve_name]
        foreign_curve = foreign_model._curves_dict[foreign_curve_name]

        # Extract parameters
        spot_fx = xccy_inputs["spot_fx"]
        tenor_list = xccy_inputs["tenor_list"]
        basis_spreads = xccy_inputs["basis_spreads"]
        domestic_freq_type = xccy_inputs["domestic_freq_type"]
        foreign_freq_type = xccy_inputs["foreign_freq_type"]
        domestic_dc_type = xccy_inputs["domestic_dc_type"]
        foreign_dc_type = xccy_inputs["foreign_dc_type"]
        interp_type = xccy_inputs["interp_type"]

        # Extract currency codes from curve names
        domestic_currency_code = domestic_curve_name.split('_')[0]
        foreign_currency_code = foreign_curve_name.split('_')[0]

        domestic_currency = CurrencyTypes[domestic_currency_code]
        foreign_currency = CurrencyTypes[foreign_currency_code]

        domestic_index = CurveTypes[domestic_curve_name]
        foreign_index = CurveTypes[foreign_curve_name]

        # Create calibration swaps (matching manual approach exactly)
        domestic_notional = 100_000_000
        foreign_notional = domestic_notional / spot_fx

        xccy_basis_spreads = [s/10000 for s in basis_spreads]
        calib_swaps = []
        for tenor, spread in zip(tenor_list, xccy_basis_spreads):
            swap = XccyBasisSwap(
                effective_dt=self.value_dt,
                term_dt_or_tenor=tenor,
                domestic_notional=domestic_notional,
                foreign_notional=foreign_notional,
                domestic_spread=0.0,
                foreign_spread=spread,
                domestic_freq_type=domestic_freq_type,
                foreign_freq_type=foreign_freq_type,
                domestic_dc_type=domestic_dc_type,
                foreign_dc_type=foreign_dc_type,
                domestic_floating_index=domestic_index,
                foreign_floating_index=foreign_index,
                domestic_currency=domestic_currency,
                foreign_currency=foreign_currency
            )
            calib_swaps.append(swap)

        # Build XCCY curve (matching manual approach exactly)
        # NOTE: Force FLAT_FWD_RATES for XCCY curves (LINEAR_ZERO_RATES causes NaN in JAX AD)
        xccy_curve = XccyCurve(
            value_dt=self.value_dt,
            basis_swaps=calib_swaps,
            domestic_curve=domestic_curve,
            foreign_curve=foreign_curve,
            spot_fx=1/spot_fx,  # XccyCurve expects USD/GBP (inverse)
            interp_type=InterpTypes.FLAT_FWD_RATES,
            use_ad=True
        )

        # Store curves in main model (matching manual approach)
        self._curves_dict[domestic_curve_name] = domestic_curve
        self._curves_dict[foreign_curve_name] = foreign_curve

        # Determine stored XCCY curve name
        foreign_ccy = xccy_inputs["fx_pair"][:3]
        domestic_ccy = xccy_inputs["fx_pair"][3:]
        stored_name = f"{foreign_ccy}_{domestic_ccy}_BASIS"
        self._curves_dict[stored_name] = xccy_curve

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