##############################################################################
# Cross-Currency (XCCY) Curve Bootstrapping Implementation
# Based on cashflow-based approach without iterative solvers
##############################################################################

import numpy as np
from cavour.utils.error import LibError
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes, DayCount
from cavour.utils.frequency import FrequencyTypes
from cavour.utils.global_types import SwapTypes, CurveTypes
from cavour.utils.currency import CurrencyTypes
from cavour.utils.global_vars import gDaysInYear
from cavour.market.curves.discount_curve import DiscountCurve
from cavour.trades.rates.swap_float_leg import SwapFloatLeg
from cavour.utils.calendar import CalendarTypes,  DateGenRuleTypes

##############################################################################


class XCCYCurve:
    """Cross-currency curve that discounts cashflows in the target currency
    using basis spreads to account for cross-currency basis risk.
    
    The curve is bootstrapped using a cashflow-based approach that eliminates
    the need for iterative solvers by directly calculating discount factors.
    """
    
    def __init__(self,
                 value_dt: Date,
                 target_ois_curve: DiscountCurve,
                 collateral_ois_curve: DiscountCurve,
                 basis_tenors: list,
                 basis_spreads: list,
                 fx_rate: float,
                 target_currency: CurrencyTypes = CurrencyTypes.GBP,
                 collateral_currency: CurrencyTypes = CurrencyTypes.USD,
                 target_index: CurveTypes = CurveTypes.GBP_OIS_SONIA,
                 collateral_index: CurveTypes = CurveTypes.USD_OIS_SOFR,
                 target_freq_type: FrequencyTypes = FrequencyTypes.QUARTERLY,
                 collateral_freq_type: FrequencyTypes = FrequencyTypes.QUARTERLY,
                 target_dc_type: DayCountTypes = DayCountTypes.ACT_365F,
                 collateral_dc_type: DayCountTypes = DayCountTypes.ACT_360):
        """
        Initialize XCCY curve with basis spread data and underlying OIS curves.
        
        Parameters:
        -----------
        value_dt : Date
            Valuation date for the curve
        target_ois_curve : DiscountCurve
            OIS curve for the target currency
        collateral_ois_curve : DiscountCurve
            OIS curve for the collateral currency
        basis_tenors : list
            List of tenor strings (e.g., ["1Y", "2Y", "5Y"])
        basis_spreads : list
            List of basis spreads in basis points as np.float64 values (e.g., 312.5 for 312.5 bps)
        fx_rate : float
            FX rate from collateral currency to target currency (e.g., 0.79 for USD/GBP)
        target_currency : CurrencyTypes
            Currency of the target leg
        collateral_currency : CurrencyTypes
            Currency of the collateral leg
        target_index : CurveTypes
            Floating index for target currency leg
        collateral_index : CurveTypes
            Floating index for collateral currency leg
        freq_type : FrequencyTypes
            Payment frequency for both legs
        dc_type : DayCountTypes
            Day count convention for both legs
        """
        
        # Validate input lengths
        if len(basis_tenors) != len(basis_spreads):
            raise LibError("basis_tenors and basis_spreads must have the same length")
        
        self._value_dt = value_dt
        self._target_ois_curve = target_ois_curve
        self._collateral_ois_curve = collateral_ois_curve
        self._basis_tenors = basis_tenors
        self._basis_spreads = basis_spreads
        self._fx_rate = fx_rate
        self._target_currency = target_currency
        self._collateral_currency = collateral_currency
        self._target_index = target_index
        self._collateral_index = collateral_index
        self._target_freq_type = target_freq_type
        self._collateral_freq_type = collateral_freq_type
        self._target_dc_type = target_dc_type
        self._collateral_dc_type = collateral_dc_type
        
        # Bootstrap the XCCY curve
       #self._xccy_curve = self._bootstrap_xccy_curve()
    
    def _create_target_leg(self, maturity_dt: Date, basis_spread: float, notional: float = 1000000.0, notional_exchange: bool = True) -> SwapFloatLeg:
        """Create target currency floating leg with basis spread."""
        return SwapFloatLeg(
            effective_dt=self._value_dt,
            end_dt=maturity_dt,
            leg_type=SwapTypes.PAY,
            spread=basis_spread,
            freq_type=self._target_freq_type,
            dc_type=self._target_dc_type,
            cal_type = CalendarTypes.UNITED_KINGDOM,
            notional=notional,
            notional_exchange=notional_exchange,
            floating_index=self._target_index,
            currency=self._target_currency
        )
    
    def _create_collateral_leg(self, maturity_dt: Date, notional: float = 1000000.0, notional_exchange: bool = True) -> SwapFloatLeg:
        """Create collateral currency floating leg without spread."""
        return SwapFloatLeg(
            effective_dt=self._value_dt,
            end_dt=maturity_dt,
            leg_type=SwapTypes.RECEIVE,
            spread=0.0,
            freq_type=self._collateral_freq_type,
            dc_type=self._collateral_dc_type,
            cal_type = CalendarTypes.UNITED_STATES,
            notional=notional,
            notional_exchange=notional_exchange,
            floating_index=self._collateral_index,
            currency=self._collateral_currency
        )
    
    def _calculate_known_pvs(self, leg: SwapFloatLeg, discount_curve: DiscountCurve, 
                           index_curve: DiscountCurve, exclude_final: bool = True):
        """
        Calculate PV of all cashflows except the final one.
        
        Returns:
        --------
        tuple: (known_pv, final_cashflow)
            known_pv: Present value of all known cashflows
            final_cashflow: Amount of the final cashflow (not discounted)
        """
        leg_pv = 0.0
        final_payment_cf = 0.0
        
        num_payments = len(leg._payment_dts)
        end_index = num_payments - 1 if exclude_final else num_payments
        
        # Set up day count convention for index curve (same as SwapFloatLeg)
        index_day_counter = DayCount(index_curve._dc_type)
        
        # Ensure notional array is properly initialized (same logic as SwapFloatLeg lines 179-188)
        if not len(leg._notional_array):
            leg._notional_array = [leg._notional] * num_payments
        elif len(leg._notional_array) != num_payments:
            # Adjust notional array length if payment dates have been modified
            if len(leg._notional_array) < num_payments:
                # Add notional for additional payments (e.g., from notional exchange)
                leg._notional_array = [leg._notional] + leg._notional_array
            else:
                # Trim excess notional entries
                leg._notional_array = leg._notional_array[:num_payments]
        
        # Calculate PV of known cashflows
        for i in range(end_index):
            # Get cashflow details
            start_dt = leg._start_accrued_dts[i]
            end_dt = leg._end_accrued_dts[i]
            payment_dt = leg._payment_dts[i]
            pay_alpha = leg._year_fracs[i]  # Payment leg day count
            
            # Calculate forward rate using index curve day count (same as SwapFloatLeg lines 203-215)
            (index_alpha, num, _) = index_day_counter.year_frac(start_dt, end_dt)
            df_start = index_curve.df(start_dt, leg._dc_type)
            df_end = index_curve.df(end_dt, leg._dc_type)
            fwd_rate = (df_start / df_end - 1.0) / index_alpha
            
            # Calculate cashflow using payment leg day count and notional (same as SwapFloatLeg)
            cashflow = (fwd_rate + leg._spread) * pay_alpha * leg._notional_array[i]
            
            # Discount using known discount factors
            df_payment = discount_curve.df(payment_dt)
            leg_pv += cashflow * df_payment
        
        # Calculate final cashflow amount (but not its PV yet)
        if num_payments > 0:
            i = num_payments - 1
            start_dt = leg._start_accrued_dts[i]
            end_dt = leg._end_accrued_dts[i]
            pay_alpha = leg._year_fracs[i]  # Payment leg day count
            
            # Final period forward rate using index curve day count (same as SwapFloatLeg lines 203-215)
            (index_alpha, num, _) = index_day_counter.year_frac(start_dt, end_dt)
            df_start = index_curve.df(start_dt, leg._dc_type)
            df_end = index_curve.df(end_dt, leg._dc_type)
            fwd_rate = (df_start / df_end - 1.0) / index_alpha
            
            # Calculate final interest payment using payment leg day count and notional (same as SwapFloatLeg)
            final_payment_cf = (fwd_rate + leg._spread) * pay_alpha * leg._notional_array[i]
            
            # Add notional exchange if enabled (same as SwapFloatLeg line 284)
            if leg._notional_exchange:
                final_payment_cf += float(leg._notional)  # Always positive at end
        
        return leg_pv, final_payment_cf
    
    def _bootstrap_single_maturity(self, tenor: str, basis_spread: float, xccy_curve_partial: DiscountCurve):
        """
        Bootstrap a single maturity using direct cashflow approach.
        
        Parameters:
        -----------
        tenor : str
            Tenor string (e.g., "1Y", "2Y")
        basis_spread : float
            Basis spread value
        xccy_curve_partial : DiscountCurve
            Partially built XCCY curve for intermediate calculations
            
        Returns:
        --------
        tuple: (maturity_dt, df_final)
            maturity_dt: Final payment date
            df_final: Calculated discount factor for final payment
        """
        # Parse maturity from tenor string
        maturity_dt = self._value_dt.add_tenor(tenor)
        
        # Convert basis points to decimal (1 bps = 0.0001)
        basis_spread_decimal = basis_spread / 10000.0
        
        # Create floating legs with FX-adjusted notionals
        # Target leg: uses base notional (e.g., 1M GBP)
        # Collateral leg: uses FX-converted notional (e.g., 1.27M USD if fx_rate = 0.79)
        base_notional = 1000000.0  # 1M in target currency
        collateral_notional = base_notional / self._fx_rate  # Convert to collateral currency
        
        target_leg = self._create_target_leg(maturity_dt, basis_spread_decimal, base_notional)
        collateral_leg = self._create_collateral_leg(maturity_dt, collateral_notional)
        
        # Calculate known PVs (all payments except final)
        target_known_pv, target_final_cf = self._calculate_known_pvs(
            target_leg, xccy_curve_partial, self._target_ois_curve, exclude_final=True)
        
        collateral_known_pv, collateral_final_cf = self._calculate_known_pvs(
            collateral_leg, self._collateral_ois_curve, self._collateral_ois_curve, exclude_final=True)
        
        # Get collateral currency final discount factor (known from collateral OIS curve)
        final_payment_dt = target_leg._payment_dts[-1]  # Should match collateral leg
        df_collateral_final = self._collateral_ois_curve.df(final_payment_dt)
        
        # Solve for target currency final discount factor
        # For cross-currency basis swap: PV_target = PV_collateral * fx_rate
        # target_known_pv + target_final_cf * df_target_final = (collateral_known_pv + collateral_final_cf * df_collateral_final) * fx_rate
        
        collateral_total_pv = collateral_known_pv + collateral_final_cf * df_collateral_final
        rhs = collateral_total_pv * self._fx_rate  # Convert collateral PV to target currency
        lhs_known = target_known_pv
        
        if abs(target_final_cf) < 1e-12:
            raise LibError(f"Target final cashflow is zero for maturity {maturity_dt}")
        
        df_target_final = (rhs - lhs_known) / target_final_cf
        
        if df_target_final <= 0:
            raise LibError(f"Negative discount factor calculated for maturity {maturity_dt}: {df_target_final}")
        
        return final_payment_dt, df_target_final
    
    def _bootstrap_xccy_curve(self) -> DiscountCurve:
        """
        Bootstrap the complete XCCY curve using sequential cashflow approach.
        
        Returns:
        --------
        DiscountCurve: Bootstrapped cross-currency discount curve
        """
        # Initialize with value date (time = 0.0)
        xccy_dates = [self._value_dt]
        xccy_times = [0.0]
        xccy_dfs = [1.0]
        
        # Build partial curve for interpolation
        xccy_curve_partial = DiscountCurve(self._value_dt, xccy_times, np.array(xccy_dfs))
        
        # Iterate over basis spreads (assumed to be already sorted)
        for tenor, spread in zip(self._basis_tenors, self._basis_spreads):

            # Bootstrap single maturity
            maturity_dt, df_final = self._bootstrap_single_maturity(
                tenor, spread, xccy_curve_partial)
            
            # Convert date to time (year fraction from value date)
            time_years = (maturity_dt - self._value_dt) / gDaysInYear
            
            # Add to curve data
            xccy_dates.append(maturity_dt)
            xccy_times.append(time_years)
            xccy_dfs.append(df_final)

            
            # Update partial curve for next iteration using times (not dates)
            xccy_curve_partial = DiscountCurve(self._value_dt, xccy_times.copy(), np.array(xccy_dfs))
        
        # Return final bootstrapped curve using times (not dates)
        return DiscountCurve(self._value_dt, xccy_times, np.array(xccy_dfs))
    
    def df(self, dt: Date) -> float:
        """Get discount factor for a given date."""
        return self._xccy_curve.df(dt)
    
    def get_curve(self) -> DiscountCurve:
        """Return the underlying discount curve."""
        return self._xccy_curve
    
    def __repr__(self):
        s = f"XCCYCurve:\n"
        s += f"  Value Date: {self._value_dt}\n"
        s += f"  Target Currency: {self._target_currency}\n"
        s += f"  Collateral Currency: {self._collateral_currency}\n"
        s += f"  FX Rate: {self._fx_rate}\n"
        s += f"  Number of basis spreads: {len(self._basis_spreads)}\n"
        # s += f"  Frequency: {self._freq_type}\n"
        # s += f"  Day Count: {self._dc_type}\n"
        return s


##############################################################################


def bootstrap_xccy_curve(value_dt: Date,
                        target_ois_curve: DiscountCurve,
                        collateral_ois_curve: DiscountCurve,
                        basis_tenors: list,
                        basis_spreads: list,
                        fx_rate: float,
                        target_currency: CurrencyTypes = CurrencyTypes.GBP,
                        collateral_currency: CurrencyTypes = CurrencyTypes.USD,
                        target_index: CurveTypes = CurveTypes.GBP_OIS_SONIA,
                        collateral_index: CurveTypes = CurveTypes.USD_OIS_SOFR,
                        freq_type: FrequencyTypes = FrequencyTypes.QUARTERLY,
                        dc_type: DayCountTypes = DayCountTypes.ACT_360) -> DiscountCurve:
    """
    Convenience function to bootstrap XCCY curve and return the DiscountCurve directly.
    
    Parameters:
    -----------
    value_dt : Date
        Valuation date for the curve
    target_ois_curve : DiscountCurve
        OIS curve for the target currency
    collateral_ois_curve : DiscountCurve
        OIS curve for the collateral currency
    basis_tenors : list
        List of tenor strings (e.g., ["1Y", "2Y", "5Y"])
    basis_spreads : list
        List of basis spreads in basis points as np.float64 values (e.g., 312.5 for 312.5 bps)
    fx_rate : float
        FX rate from collateral currency to target currency (e.g., 0.79 for USD/GBP)
    target_currency : CurrencyTypes
        Currency of the target leg
    collateral_currency : CurrencyTypes
        Currency of the collateral leg
    target_index : CurveTypes
        Floating index for target currency leg
    collateral_index : CurveTypes
        Floating index for collateral currency leg
    freq_type : FrequencyTypes
        Payment frequency for both legs
    dc_type : DayCountTypes
        Day count convention for both legs
        
    Returns:
    --------
    DiscountCurve: Bootstrapped cross-currency discount curve
    """
    xccy_curve_builder = XCCYCurve(
        value_dt=value_dt,
        target_ois_curve=target_ois_curve,
        collateral_ois_curve=collateral_ois_curve,
        basis_tenors=basis_tenors,
        basis_spreads=basis_spreads,
        fx_rate=fx_rate,
        target_currency=target_currency,
        collateral_currency=collateral_currency,
        target_index=target_index,
        collateral_index=collateral_index,
        freq_type=freq_type,
        dc_type=dc_type
    )
    
    return xccy_curve_builder.get_curve()

##############################################################################