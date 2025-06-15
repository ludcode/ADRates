from typing import Dict, List
from dataclasses import dataclass, field

from cavour.utils import *
from cavour.trades.rates.ois_curve import OISCurve
from cavour.trades.rates.ois import OIS
from cavour.marketdata.market_data_constants import MARKET_DATA
from cavour.marketdata.market_data_engine import MarketCurveBuilder


class CurveAccessor:
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
    value_dt: Date
    _curves_dict: Dict[str, OISCurve] = field(default_factory=dict)
    _curve_params_dict: Dict[str, dict] = field(default_factory=dict)  # ← Add this line
    

    def prebuilt_curve(
        self,
        curve_names: Union[str, List[str]],
    ):
        
        builder = MarketCurveBuilder(MARKET_DATA)

        if isinstance(curve_names, str):
            curve_names = [curve_names]

        for curve_name in curve_names:
            curve_inputs = builder.get_curve_inputs(curve_name, self.value_dt)
            self.build_curve(**curve_inputs)
            


    def build_curve(
        self,
        name: str,
        px_list: List[float],
        tenor_list: List[str],
        spot_days: int = 0,
        swap_type=SwapTypes.PAY,
        fixed_dcc_type=DayCountTypes.ACT_365F,
        fixed_freq_type=FrequencyTypes.ANNUAL,
        float_freq_type=FrequencyTypes.ANNUAL,
        float_dc_type=DayCountTypes.ACT_365F,
        bus_day_type=BusDayAdjustTypes.MODIFIED_FOLLOWING,
        interp_type=InterpTypes.LINEAR_ZERO_RATES
    ):
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
                float_dc_type=float_dc_type
            )
            for tenor, px in zip(tenor_list, px_list)
        ]

        curve = OISCurve(
            value_dt=self.value_dt,
            ois_swaps=swaps,
            interp_type=interp_type,
            check_refit=False
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

    def scenario(self, curve_name: str, shock: dict | float, new_name: str | None = None):
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
        return CurveAccessor(self._curves_dict)