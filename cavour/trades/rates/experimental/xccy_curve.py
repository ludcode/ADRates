##############################################################################
# Cross-Currency (XCCY) Curve Bootstrapping Implementation
# Based on cashflow-based approach without iterative solvers
##############################################################################

from ....utils.error import LibError
from ....utils.date import Date
from ....utils.day_count import DayCountTypes
from ....utils.frequency import FrequencyTypes
from ....utils.global_types import SwapTypes, CurveTypes
from ....utils.currency import CurrencyTypes
from ....market.curves.discount_curve import DiscountCurve
from ..swap_float_leg import SwapFloatLeg

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
                 basis_spreads: list,
                 target_currency: CurrencyTypes = CurrencyTypes.GBP,
                 collateral_currency: CurrencyTypes = CurrencyTypes.USD,
                 target_index: CurveTypes = CurveTypes.GBP_OIS_SONIA,
                 collateral_index: CurveTypes = CurveTypes.USD_OIS_SOFR,
                 freq_type: FrequencyTypes = FrequencyTypes.QUARTERLY,
                 dc_type: DayCountTypes = DayCountTypes.ACT_360):
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
        basis_spreads : list
            List of dictionaries with 'maturity' and 'spread' keys
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
        
        self._value_dt = value_dt
        self._target_ois_curve = target_ois_curve
        self._collateral_ois_curve = collateral_ois_curve
        self._basis_spreads = basis_spreads
        self._target_currency = target_currency
        self._collateral_currency = collateral_currency
        self._target_index = target_index
        self._collateral_index = collateral_index
        self._freq_type = freq_type
        self._dc_type = dc_type
        
        # Bootstrap the XCCY curve
        self._xccy_curve = self._bootstrap_xccy_curve()
    
    def _create_target_leg(self, maturity_dt: Date, basis_spread: float) -> SwapFloatLeg:
        """Create target currency floating leg with basis spread."""
        return SwapFloatLeg(
            effective_dt=self._value_dt,
            end_dt=maturity_dt,
            leg_type=SwapTypes.PAY,
            spread=basis_spread,
            freq_type=self._freq_type,
            dc_type=self._dc_type,
            floating_index=self._target_index,
            currency=self._target_currency
        )
    
    def _create_collateral_leg(self, maturity_dt: Date) -> SwapFloatLeg:
        """Create collateral currency floating leg without spread."""
        return SwapFloatLeg(
            effective_dt=self._value_dt,
            end_dt=maturity_dt,
            leg_type=SwapTypes.RECEIVE,
            spread=0.0,
            freq_type=self._freq_type,
            dc_type=self._dc_type,
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
        
        # Calculate PV of known cashflows
        for i in range(end_index):
            # Get cashflow details
            start_dt = leg._start_accrued_dts[i]
            end_dt = leg._end_accrued_dts[i]
            payment_dt = leg._payment_dts[i]
            year_frac = leg._year_fracs[i]
            
            # Calculate forward rate
            df_start = index_curve.df(start_dt)
            df_end = index_curve.df(end_dt)
            index_alpha = year_frac  # Using payment leg day count
            fwd_rate = (df_start / df_end - 1.0) / index_alpha
            
            # Calculate cashflow
            cashflow = (fwd_rate + leg._spread) * year_frac * leg._notional
            
            # Discount using known discount factors
            df_payment = discount_curve.df(payment_dt)
            leg_pv += cashflow * df_payment
        
        # Calculate final cashflow amount (but not its PV yet)
        if num_payments > 0:
            i = num_payments - 1
            start_dt = leg._start_accrued_dts[i]
            end_dt = leg._end_accrued_dts[i]
            year_frac = leg._year_fracs[i]
            
            # Final period forward rate
            df_start = index_curve.df(start_dt)
            df_end = index_curve.df(end_dt)
            index_alpha = year_frac
            fwd_rate = (df_start / df_end - 1.0) / index_alpha
            
            final_payment_cf = (fwd_rate + leg._spread) * year_frac * leg._notional
        
        return leg_pv, final_payment_cf
    
    def _bootstrap_single_maturity(self, spread_data: dict, xccy_curve_partial: DiscountCurve):
        """
        Bootstrap a single maturity using direct cashflow approach.
        
        Parameters:
        -----------
        spread_data : dict
            Dictionary containing 'maturity' (Date or tenor string) and 'spread' (float)
        xccy_curve_partial : DiscountCurve
            Partially built XCCY curve for intermediate calculations
            
        Returns:
        --------
        tuple: (maturity_dt, df_final)
            maturity_dt: Final payment date
            df_final: Calculated discount factor for final payment
        """
        # Parse maturity
        if isinstance(spread_data["maturity"], str):
            maturity_dt = self._value_dt.add_tenor(spread_data["maturity"])
        else:
            maturity_dt = spread_data["maturity"]
            
        basis_spread = spread_data["spread"]
        
        # Create floating legs
        target_leg = self._create_target_leg(maturity_dt, basis_spread)
        collateral_leg = self._create_collateral_leg(maturity_dt)
        
        # Calculate known PVs (all payments except final)
        target_known_pv, target_final_cf = self._calculate_known_pvs(
            target_leg, xccy_curve_partial, self._target_ois_curve, exclude_final=True)
        
        collateral_known_pv, collateral_final_cf = self._calculate_known_pvs(
            collateral_leg, self._collateral_ois_curve, self._collateral_ois_curve, exclude_final=True)
        
        # Get collateral currency final discount factor (known from collateral OIS curve)
        final_payment_dt = target_leg._payment_dts[-1]  # Should match collateral leg
        df_collateral_final = self._collateral_ois_curve.df(final_payment_dt)
        
        # Solve for target currency final discount factor
        # For cross-currency basis swap: PV_target = PV_collateral
        # target_known_pv + target_final_cf * df_target_final = collateral_known_pv + collateral_final_cf * df_collateral_final
        
        rhs = collateral_known_pv + collateral_final_cf * df_collateral_final
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
        # Initialize with value date
        xccy_dates = [self._value_dt]
        xccy_dfs = [1.0]
        
        # Build partial curve for interpolation
        xccy_curve_partial = DiscountCurve(self._value_dt, xccy_dates, xccy_dfs)
        
        # Sort basis spreads by maturity to ensure sequential bootstrap
        sorted_spreads = sorted(self._basis_spreads, 
                              key=lambda x: x["maturity"] if isinstance(x["maturity"], Date) 
                              else self._value_dt.add_tenor(x["maturity"]))
        
        for spread_data in sorted_spreads:
            # Bootstrap single maturity
            maturity_dt, df_final = self._bootstrap_single_maturity(
                spread_data, xccy_curve_partial)
            
            # Add to curve
            xccy_dates.append(maturity_dt)
            xccy_dfs.append(df_final)
            
            # Update partial curve for next iteration
            xccy_curve_partial = DiscountCurve(self._value_dt, xccy_dates.copy(), xccy_dfs.copy())
        
        # Return final bootstrapped curve
        return DiscountCurve(self._value_dt, xccy_dates, xccy_dfs)
    
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
        s += f"  Number of basis spreads: {len(self._basis_spreads)}\n"
        s += f"  Frequency: {self._freq_type}\n"
        s += f"  Day Count: {self._dc_type}\n"
        return s


##############################################################################


def bootstrap_xccy_curve(value_dt: Date,
                        target_ois_curve: DiscountCurve,
                        collateral_ois_curve: DiscountCurve,
                        basis_spreads: list,
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
    basis_spreads : list
        List of dictionaries with 'maturity' and 'spread' keys
        Example: [{"maturity": "3M", "spread": 0.0015}, {"maturity": "1Y", "spread": 0.0020}]
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
        basis_spreads=basis_spreads,
        target_currency=target_currency,
        collateral_currency=collateral_currency,
        target_index=target_index,
        collateral_index=collateral_index,
        freq_type=freq_type,
        dc_type=dc_type
    )
    
    return xccy_curve_builder.get_curve()

##############################################################################