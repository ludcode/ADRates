"Valuation Engine"

import numpy as np
import jax.numpy as jnp
from jax import lax, jit, grad, hessian, jacrev, linearize
from functools import partial
from typing import Sequence, Any, Dict

from cavour.market.curves.interpolator import *
from cavour.utils.helpers import to_tenor, times_from_dates
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.error import LibError
from cavour.market.curves.interpolator_ad import InterpolatorAd
from cavour.requests.results import Valuation, Gamma, Delta, AnalyticsResult, Risk, CrossGamma, Cashflows, CashflowItem
from cavour.utils.global_types import (SwapTypes,
                                   InstrumentTypes,
                                   RequestTypes,
                                   CurveTypes)
from cavour.utils.currency import CurrencyTypes
from cavour.trades.rates.swap_fixed_leg import SwapFixedLeg
from cavour.trades.rates.swap_float_leg import SwapFloatLeg



class Engine:
    def __init__(self,
                 model):

        self.model = model
        # cache bootstrapped curves keyed by curve name
        self._curve_cache: Dict[Any, Dict[str, Any]] = {}

    def _extract_leg_cashflows(self, leg, leg_type_str: str) -> list:
        """
        Extract cashflows from a swap leg after value() has been called.

        Args:
            leg: SwapFixedLeg or SwapFloatLeg instance (must have been valued)
            leg_type_str: String like "Fixed_Pay", "Float_Rec", "Notional_Pay", etc.

        Returns:
            List of CashflowItem objects
        """
        cashflow_items = []

        # Check if leg has been valued (has payment_dfs attribute)
        if not hasattr(leg, '_payment_dfs') or not leg._payment_dfs:
            return cashflow_items

        # Determine sign based on pay/receive
        # Pay legs are negative cashflows from our perspective, Receive legs are positive
        sign = -1.0 if 'Pay' in leg_type_str else 1.0

        num_payments = len(leg._payment_dts)

        for i in range(num_payments):
            # Get notional for this payment (handle notional arrays for floating legs)
            if hasattr(leg, '_notional_array') and leg._notional_array:
                notional = float(leg._notional_array[i]) if i < len(leg._notional_array) else float(leg._notional)
            else:
                notional = float(leg._notional)

            # Calculate payment fraction (rate * year_frac for fixed, or just the computed rate for float)
            if notional != 0:
                payment_fraction = float(leg._payments[i]) / notional
            else:
                payment_fraction = 0.0

            # Apply sign convention: Pay = negative, Receive = positive
            signed_amount = sign * float(leg._payments[i])
            signed_pv = sign * float(leg._payment_pvs[i])

            # Create cashflow item
            cf_item = CashflowItem(
                payment_date=leg._payment_dts[i],
                notional=notional,
                payment_fraction=payment_fraction,
                accrual_period=float(leg._year_fracs[i]),
                amount=signed_amount,
                discount_factor=float(leg._payment_dfs[i]),
                discounted_amount=signed_pv,
                leg_type=leg_type_str
            )
            cashflow_items.append(cf_item)

        return cashflow_items

    def compute(self, derivative, request_list, collateral_type=None):
        """Return analytics for the given derivative and requested measures.

        Args:
            derivative: Derivative instrument (OIS, XCCY swap, etc.)
            request_list: List of RequestTypes (VALUE, DELTA, GAMMA)
            collateral_type (CollateralType, optional): Type of collateral for discounting.
                If None, uses natural currency. For cross-currency collateral, specify
                the collateral currency (e.g., CollateralType.USD).

        Returns:
            AnalyticsResult with value, risk (delta), and gamma
        """
        reqs = set(request_list)

        # Route XCCY swaps to separate handler
        if derivative.derivative_type == InstrumentTypes.XCCY_SWAP:
            return self._compute_xccy(derivative, reqs, collateral_type)

        # Route OIS swaps to new handler (supports collateral)
        if derivative.derivative_type == InstrumentTypes.OIS_SWAP:
            return self._compute_ois(derivative, reqs, collateral_type)

        # Route bonds to bond handler
        if derivative.derivative_type == InstrumentTypes.BOND:
            return self._compute_bond(derivative, reqs)

        # Route FRNs to FRN handler
        if derivative.derivative_type == InstrumentTypes.FRN:
            return self._compute_frn(derivative, reqs)

        # Route YoY Inflation Swaps to YoY IIS handler
        if derivative.derivative_type == InstrumentTypes.YOY_INFLATION_SWAP:
            return self._compute_yoy_iis(derivative, reqs)

        raise LibError(f"{derivative.derivative_type} not yet implemented")

    def _compute_ois(self, derivative, reqs, collateral_type=None):
        """Compute analytics for OIS swaps with optional cross-currency collateral.

        Args:
            derivative: OIS swap instance
            reqs: Set of RequestTypes (VALUE, DELTA, GAMMA)
            collateral_type (CollateralType, optional): Collateral currency

        Returns:
            AnalyticsResult with value, risk (delta), and gamma
        """
        from cavour.utils.global_types import collateral_to_currency

        # Determine collateral currency
        if collateral_type is None:
            collateral_ccy = derivative._currency  # Natural currency (default)
        else:
            collateral_ccy = collateral_to_currency(collateral_type)

        # Natural currency path (existing single-curve logic)
        if collateral_ccy == derivative._currency:
            return self._compute_ois_natural(derivative, reqs)

        # Cross-currency collateral path (new dual-curve logic)
        else:
            return self._compute_ois_xccy_collateral(derivative, reqs, collateral_ccy)

    def _compute_ois_natural(self, derivative, reqs):
        """Compute OIS with natural currency (single-curve discounting)."""
        ir_model = getattr(self.model.curves, derivative._floating_index.name)

        fixed = self._fixed_leg_analytics(
            ir_model.swap_rates,
            ir_model.swap_times,
            ir_model.year_fracs,
            derivative._fixed_leg,
            ir_model._value_dt,
            ir_model._interp_type,
            reqs,
        )

        floating = self._float_leg_analytics(
            ir_model.swap_rates,
            ir_model.swap_times,
            ir_model.year_fracs,
            derivative._float_leg,
            ir_model._value_dt,
            ir_model._interp_type,
            ir_model._interp_type,
            None,
            reqs,
        )

        value = None
        if RequestTypes.VALUE in reqs:
            value = fixed.get("value") + floating.get("value")

        delta = None
        if RequestTypes.DELTA in reqs:
            delta = fixed.get("delta") + floating.get("delta")

        gamma = None
        if RequestTypes.GAMMA in reqs:
            gamma = fixed.get("gamma") + floating.get("gamma")

        # Cashflows extraction
        cashflows = None
        if RequestTypes.CASHFLOWS in reqs:
            all_cashflows = []

            # Value the legs explicitly to populate cashflow data
            # (The analytics methods use separate copies for AD, so we need to value the original legs)
            derivative._fixed_leg.value(ir_model._value_dt, ir_model)
            derivative._float_leg.value(ir_model._value_dt, ir_model, ir_model)

            # Determine leg types based on swap direction
            fixed_leg_type = "Fixed_Pay" if derivative._fixed_leg._leg_type == SwapTypes.PAY else "Fixed_Rec"
            float_leg_type = "Float_Rec" if derivative._fixed_leg._leg_type == SwapTypes.PAY else "Float_Pay"

            # Extract cashflows from fixed leg
            fixed_cfs = self._extract_leg_cashflows(derivative._fixed_leg, fixed_leg_type)
            all_cashflows.extend(fixed_cfs)

            # Extract cashflows from floating leg
            float_cfs = self._extract_leg_cashflows(derivative._float_leg, float_leg_type)
            all_cashflows.extend(float_cfs)

            cashflows = Cashflows(all_cashflows, derivative._currency)

        return AnalyticsResult(value=value, risk=delta, gamma=gamma, cashflows=cashflows)

    def _compute_ois_xccy_collateral(self, derivative, reqs, collateral_ccy):
        """Compute OIS with cross-currency collateral (dual-curve discounting).

        Uses OIS curve for forward rate projection and XCCY curve for discounting.
        Follows the same pattern as _compute_xccy() for dual-curve valuation.

        Phase 1: VALUE support only (DELTA and GAMMA to be added later).
        """
        import jax.numpy as jnp
        from cavour.utils.helpers import times_from_dates

        # Get OIS curve for projection
        ois_model = getattr(self.model.curves, derivative._floating_index.name)

        # Get XCCY curve for discounting
        swap_ccy_code = derivative._currency.name
        collateral_ccy_code = collateral_ccy.name
        xccy_curve_name = f"{swap_ccy_code}_{collateral_ccy_code}_XCCY"

        try:
            xccy_curve = getattr(self.model.curves, xccy_curve_name)
            spot_fx = xccy_curve._spot_fx
        except AttributeError:
            raise LibError(f"XCCY curve {xccy_curve_name} not found in model. "
                         f"Required for cross-currency collateral valuation. "
                         f"Available curves: {[attr for attr in dir(self.model.curves) if not attr.startswith('_')]}")

        # Build OIS curve arrays (for forward rate projection)
        ois_curve_key = tuple(ois_model.swap_times)
        ois_cache = self._cached_curve(
            ois_curve_key,
            ois_model.swap_rates,
            ois_model.swap_times,
            ois_model.year_fracs,
            ois_model._interp_type
        )
        ois_times = ois_cache["times"]
        ois_dfs = ois_cache["dfs"]

        # Build XCCY curve arrays (for discounting)
        xccy_times = jnp.array(xccy_curve._times)
        xccy_dfs = jnp.array(xccy_curve._dfs)

        # Prepare leg parameters for JAX computation
        # Fixed leg uses XCCY curve for discounting
        # Float leg uses XCCY curve for discounting, OIS curve for forward rates

        dc_type = derivative._fixed_leg._dc_type
        value_time = times_from_dates(self.model.value_dt, self.model.value_dt, dc_type)

        # Fixed leg parameters
        fixed_payment_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                         for dt in derivative._fixed_leg._payment_dts])
        fixed_alphas = jnp.array(derivative._fixed_leg._year_fracs)
        fixed_coupons = jnp.full_like(fixed_alphas, derivative._fixed_leg._cpn)
        # SwapFixedLeg doesn't have _notional_array, only _notional
        fixed_notionals = jnp.full_like(fixed_alphas, derivative._fixed_leg._notional)
        fixed_principal = derivative._fixed_leg._principal
        fixed_leg_sign = +1.0 if derivative._fixed_leg._leg_type == SwapTypes.RECEIVE else -1.0

        # Float leg parameters
        float_payment_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                         for dt in derivative._float_leg._payment_dts])
        float_start_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                       for dt in derivative._float_leg._start_accrued_dts])
        float_end_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                     for dt in derivative._float_leg._end_accrued_dts])
        float_alphas = jnp.array(derivative._float_leg._year_fracs)
        float_spreads = jnp.full_like(float_alphas, derivative._float_leg._spread)
        float_notionals = jnp.array(derivative._float_leg._notional_array or
                                    [derivative._float_leg._notional] * len(float_alphas))
        float_principal = derivative._float_leg._principal
        float_leg_sign = +1.0 if derivative._float_leg._leg_type == SwapTypes.RECEIVE else -1.0

        # Compute VALUE using JAX
        value = None
        if RequestTypes.VALUE in reqs:
            # Fixed leg PV (discounted on XCCY curve)
            # Compute fixed leg payments: coupon * alpha * notional
            fixed_payments = fixed_coupons * fixed_alphas * fixed_notionals

            fixed_pv = self._price_fixed_leg_jax(
                dfs=xccy_dfs,
                times=xccy_times,
                interp_type=xccy_curve._interp_type,
                payment_times=fixed_payment_times,
                payments=fixed_payments,
                principal=fixed_principal,
                notional=derivative._fixed_leg._notional,
                leg_sign=fixed_leg_sign,
                value_time=value_time
            )

            # Float leg PV (dual curve: XCCY for discount, OIS for index)
            float_pv = self._float_leg_jax(
                dfs=xccy_dfs,  # XCCY curve for discounting
                times=xccy_times,
                disc_interp_type=xccy_curve._interp_type,
                idx_interp_type=ois_model._interp_type,
                payment_times=float_payment_times,
                start_times=float_start_times,
                end_times=float_end_times,
                pay_alphas=float_alphas,
                spreads=float_spreads,
                notionals=float_notionals,
                principal=float_principal,
                leg_sign=float_leg_sign,
                value_time=value_time,
                first_fixing_rate=0.0,
                override_first=False,
                idx_times=ois_times,  # OIS curve for forward rates
                idx_dfs=ois_dfs,
                notional_exchange=False,
                notional_exchange_amount=0.0,
                effective_time=value_time,
                maturity_time=value_time
            )

            # Convert to scalars and compute total PV
            fixed_pv_scalar = float(jnp.squeeze(fixed_pv))
            float_pv_scalar = float(jnp.squeeze(float_pv))
            total_pv_swap_ccy = fixed_pv_scalar + float_pv_scalar

            # Convert to collateral currency
            total_pv_collateral_ccy = total_pv_swap_ccy / spot_fx

            value = Valuation(amount=total_pv_collateral_ccy, currency=collateral_ccy)

        # Define PV functions for gradient computation (used by DELTA)
        # These functions compute PV as a function of different curve variables

        # OIS curve: affects float leg forward rates (XCCY curve fixed)
        def pv_ois_fn(ois_dfs_var):
            return self._float_leg_jax(
                dfs=xccy_dfs,  # XCCY curve for discounting (FIXED)
                times=xccy_times,
                disc_interp_type=xccy_curve._interp_type,
                idx_interp_type=ois_model._interp_type,
                payment_times=float_payment_times,
                start_times=float_start_times,
                end_times=float_end_times,
                pay_alphas=float_alphas,
                spreads=float_spreads,
                notionals=float_notionals,
                principal=float_principal,
                leg_sign=float_leg_sign,
                value_time=value_time,
                first_fixing_rate=0.0,
                override_first=False,
                idx_times=ois_times,
                idx_dfs=ois_dfs_var,  # OIS DFs (VARIABLE)
                notional_exchange=False,
                notional_exchange_amount=0.0,
                effective_time=value_time,
                maturity_time=value_time
            )

        # XCCY curve: affects both fixed and float leg discounting (OIS curve fixed)
        def pv_xccy_fn(xccy_dfs_var):
            # Fixed leg PV
            fixed_payments = fixed_coupons * fixed_alphas * fixed_notionals
            fixed_pv = self._price_fixed_leg_jax(
                dfs=xccy_dfs_var,  # XCCY DFs (VARIABLE)
                times=xccy_times,
                interp_type=xccy_curve._interp_type,
                payment_times=fixed_payment_times,
                payments=fixed_payments,
                principal=fixed_principal,
                notional=derivative._fixed_leg._notional,
                leg_sign=fixed_leg_sign,
                value_time=value_time
            )

            # Float leg PV
            float_pv = self._float_leg_jax(
                dfs=xccy_dfs_var,  # XCCY DFs (VARIABLE)
                times=xccy_times,
                disc_interp_type=xccy_curve._interp_type,
                idx_interp_type=ois_model._interp_type,
                payment_times=float_payment_times,
                start_times=float_start_times,
                end_times=float_end_times,
                pay_alphas=float_alphas,
                spreads=float_spreads,
                notionals=float_notionals,
                principal=float_principal,
                leg_sign=float_leg_sign,
                value_time=value_time,
                first_fixing_rate=0.0,
                override_first=False,
                idx_times=ois_times,
                idx_dfs=ois_dfs,  # OIS DFs (FIXED)
                notional_exchange=False,
                notional_exchange_amount=0.0,
                effective_time=value_time,
                maturity_time=value_time
            )

            return fixed_pv + float_pv

        # Wrapper functions for "original" DFs (excluding prepended t≈0)
        def pv_ois_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs])
            return pv_ois_fn(full_dfs)

        def pv_xccy_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs])
            return pv_xccy_fn(full_dfs)

        # Compute DELTA using automatic differentiation
        delta = None
        if RequestTypes.DELTA in reqs:
            from jax import grad
            from cavour.utils.helpers import to_tenor

            # Extract original DFs (excluding prepended t≈0 if present)
            ois_dfs_original = ois_dfs[1:] if ois_times[0] < 1e-6 else ois_dfs
            xccy_dfs_original = xccy_dfs[1:] if xccy_times[0] < 1e-6 else xccy_dfs

            # Compute gradients w.r.t. ORIGINAL DFs only (excluding DF(0)=1.0)
            grad_ois_dfs_original = grad(lambda d: jnp.squeeze(pv_ois_original_dfs(d)))(ois_dfs_original)
            grad_xccy_dfs_original = grad(lambda d: jnp.squeeze(pv_xccy_original_dfs(d)))(xccy_dfs_original)

            # Chain rule: OIS curve sensitivities to rates
            # Get Jacobian from cached curve
            jac_ois_original = ois_cache["jac"][1:, :] if ois_times[0] < 1e-6 else ois_cache["jac"]
            delta_ois_rates_raw = jnp.dot(grad_ois_dfs_original, jac_ois_original)

            # Convert to collateral currency and bp units
            # PV is in swap currency (GBP), convert to collateral currency (USD)
            # Then multiply by 1e-4 for bp units
            delta_ois_rates = [float(x) / spot_fx * 1e-4 for x in delta_ois_rates_raw]

            # Chain rule: XCCY curve sensitivities to basis spreads
            # Check if JAX-based bootstrap was used (has _jac_basis attribute)
            if hasattr(xccy_curve, '_jac_basis') and xccy_curve._jac_basis is not None:
                # Get basis spread tenors
                basis_swap_tenors = to_tenor(xccy_curve.swap_times)

                # Get Jacobian
                jac_xccy_pillar = xccy_curve._jac_basis[1:, :] if xccy_times[0] < 1e-6 else xccy_curve._jac_basis

                # Compute delta
                delta_xccy_rates_raw = jnp.dot(grad_xccy_dfs_original, jac_xccy_pillar)

                # Convert to collateral currency and bp units
                delta_xccy_rates = [float(x) / spot_fx * 1e-4 for x in delta_xccy_rates_raw]

                delta_xccy = Delta(
                    risk_ladder=delta_xccy_rates,
                    tenors=basis_swap_tenors,
                    currency=collateral_ccy,
                    curve_type=CurveTypes.USD_GBP_BASIS,
                )
            else:
                # Fallback: no XCCY delta if Jacobian not available
                delta_xccy = None

            # Create Delta objects for each curve
            delta_ois = Delta(
                risk_ladder=delta_ois_rates,
                tenors=to_tenor(ois_model.swap_times),
                currency=collateral_ccy,
                curve_type=derivative._floating_index,
            )

            # Package deltas into Risk object
            if delta_xccy is not None:
                delta = Risk([delta_ois, delta_xccy])
            else:
                delta = Risk([delta_ois])

        # GAMMA not yet implemented for cross-currency collateral
        gamma = None
        if RequestTypes.GAMMA in reqs:
            raise NotImplementedError(
                "GAMMA not yet supported for OIS with cross-currency collateral. "
                "Only VALUE and DELTA are currently implemented."
            )

        # Cashflows extraction (placeholder for future implementation)
        cashflows = None
        if RequestTypes.CASHFLOWS in reqs:
            # TODO: Extract cashflow data from fixed and floating legs
            cashflows = Cashflows([], derivative._currency)

        return AnalyticsResult(value=value, risk=delta, gamma=gamma, cashflows=cashflows)

    def _compute_bond(self, derivative, reqs):
        """Compute analytics for bonds (VALUE, DELTA, GAMMA).

        Args:
            derivative: Bond instance
            reqs: Set of RequestTypes (VALUE, DELTA, GAMMA)

        Returns:
            AnalyticsResult with value, risk (delta), and gamma
        """
        # Get the curve name from the bond's currency
        # Bonds discount on the OIS curve for their currency
        curve_name_map = {
            CurrencyTypes.GBP: "GBP_OIS_SONIA",
            CurrencyTypes.USD: "USD_OIS_SOFR",
            CurrencyTypes.EUR: "EUR_OIS_ESTR",
        }

        if derivative._currency not in curve_name_map:
            raise LibError(f"No default OIS curve for currency {derivative._currency}")

        curve_name = curve_name_map[derivative._currency]
        ir_model = getattr(self.model.curves, curve_name)

        # Get cached curve data
        curve_key = tuple(ir_model.swap_times)
        cache = self._cached_curve(
            curve_key,
            ir_model.swap_rates,
            ir_model.swap_times,
            ir_model.year_fracs,
            ir_model._interp_type
        )

        times = cache["times"]
        dfs = cache["dfs"]
        jac = cache["jac"]
        hess_curve = cache["hess"]

        # Extract bond cashflow data
        dc_type = derivative._dc_type
        value_dt = ir_model._value_dt

        # Convert payment dates to times
        payment_times = jnp.array(
            [times_from_dates(dt, value_dt, dc_type) for dt in derivative._payment_dts]
        )

        # Get coupon payments
        payments = jnp.array(derivative._coupon_payments)

        # Principal and notional
        principal = derivative._face_value
        notional = derivative._face_value

        # Bonds are always "receive" from investor perspective
        leg_sign = +1.0

        # Value time
        value_time = times_from_dates(value_dt, value_dt, dc_type)

        # Create partial function for bond pricing
        pv_fn = partial(
            self._price_fixed_leg_jax,
            times=times,
            interp_type=ir_model._interp_type,
            payment_times=payment_times,
            payments=payments,
            principal=principal,
            notional=notional,
            leg_sign=leg_sign,
            value_time=value_time,
        )

        # Initialize results
        value = None
        delta = None
        gamma = None

        # Compute VALUE
        if RequestTypes.VALUE in reqs:
            val = pv_fn(dfs)
            # Convert to scalar
            val_scalar = float(jnp.atleast_1d(val).item() if jnp.ndim(val) == 0 else val.squeeze())
            value = Valuation(amount=val_scalar, currency=derivative._currency)

        # Compute gradient (needed for both DELTA and GAMMA)
        need_grad = RequestTypes.DELTA in reqs or RequestTypes.GAMMA in reqs
        grad_dfs = None
        if need_grad:
            grad_dfs = grad(lambda d: jnp.squeeze(pv_fn(d)))(dfs)

        # Compute DELTA
        if RequestTypes.DELTA in reqs:
            # Chain rule: sensitivity to rates = d(PV)/d(DFs) * d(DFs)/d(rates)
            sensitivities = jnp.dot(grad_dfs, jac)
            # Convert to bp and extract values
            sensies = [float(x) * 1e-4 for x in sensitivities]

            # Create curve type enum based on currency
            curve_type_map = {
                CurrencyTypes.GBP: CurveTypes.GBP_OIS_SONIA,
                CurrencyTypes.USD: CurveTypes.USD_OIS_SOFR,
                CurrencyTypes.EUR: CurveTypes.EUR_OIS_ESTR,
            }
            curve_type = curve_type_map.get(derivative._currency, CurveTypes.GBP_OIS_SONIA)

            delta = Delta(
                risk_ladder=sensies,
                tenors=to_tenor(ir_model.swap_times),
                currency=derivative._currency,
                curve_type=curve_type,
            )

        # Compute GAMMA
        if RequestTypes.GAMMA in reqs:
            # Compute Hessian w.r.t. discount factors
            hess_dfs = hessian(lambda d: jnp.squeeze(pv_fn(d)))(dfs)

            # Chain rule for second derivatives:
            # d²PV/dr² = d²PV/dDF² * (dDF/dr)² + dPV/dDF * d²DF/dr²
            term1 = jac.T @ hess_dfs @ jac  # First term
            term2 = jnp.sum(grad_dfs[:, None, None] * hess_curve, axis=0)  # Second term
            gammas = term1 + term2

            # Convert to bp²
            gammas = np.array(gammas, dtype=np.float64) * 1e-8

            # Create curve type enum
            curve_type_map = {
                CurrencyTypes.GBP: CurveTypes.GBP_OIS_SONIA,
                CurrencyTypes.USD: CurveTypes.USD_OIS_SOFR,
                CurrencyTypes.EUR: CurveTypes.EUR_OIS_ESTR,
            }
            curve_type = curve_type_map.get(derivative._currency, CurveTypes.GBP_OIS_SONIA)

            gamma = Gamma(
                risk_ladder=gammas,
                tenors=to_tenor(ir_model.swap_times),
                currency=derivative._currency,
                curve_type=curve_type,
            )

        # Cashflows extraction
        cashflows = None
        if RequestTypes.CASHFLOWS in reqs:
            all_cashflows = []

            # Value the bond to populate cashflow data
            derivative.value(ir_model._value_dt, ir_model)

            # Extract coupon and principal cashflows
            num_payments = len(derivative._payment_dts)

            for i in range(num_payments):
                payment_dt = derivative._payment_dts[i]
                coupon_amt = derivative._coupon_payments[i]
                principal_amt = derivative._principal_payments[i] if hasattr(derivative, '_principal_payments') else 0.0

                # Add coupon cashflow if non-zero
                if abs(coupon_amt) > 1e-10:
                    # Calculate coupon rate from payment amount and notional
                    notional = derivative._principal_schedule[i] if hasattr(derivative, '_principal_schedule') else derivative._face_value
                    coupon_fraction = coupon_amt / notional if notional != 0 else 0.0

                    cf_item = CashflowItem(
                        payment_date=payment_dt,
                        notional=notional,
                        payment_fraction=coupon_fraction,
                        accrual_period=float(derivative._year_fracs[i]),
                        amount=float(coupon_amt),
                        discount_factor=float(derivative._payment_dfs[i]),
                        discounted_amount=float(derivative._coupon_pvs[i]),
                        leg_type="Coupon"
                    )
                    all_cashflows.append(cf_item)

                # Add principal cashflow if non-zero
                if abs(principal_amt) > 1e-10:
                    cf_item = CashflowItem(
                        payment_date=payment_dt,
                        notional=principal_amt,
                        payment_fraction=1.0,  # Principal repayment
                        accrual_period=0.0,  # No accrual for principal
                        amount=float(principal_amt),
                        discount_factor=float(derivative._payment_dfs[i]),
                        discounted_amount=float(derivative._principal_pvs[i]),
                        leg_type="Principal"
                    )
                    all_cashflows.append(cf_item)

            cashflows = Cashflows(all_cashflows, derivative._currency)

        return AnalyticsResult(value=value, risk=delta, gamma=gamma, cashflows=cashflows)

    def _compute_frn(self, derivative, reqs):
        """Compute analytics for FRNs (Floating Rate Notes) with VALUE, DELTA, GAMMA.

        Args:
            derivative: FRN instance
            reqs: Set of RequestTypes (VALUE, DELTA, GAMMA)

        Returns:
            AnalyticsResult with value, risk (delta), and gamma
        """
        # Get the discount curve from the FRN's currency
        curve_name_map = {
            CurrencyTypes.GBP: "GBP_OIS_SONIA",
            CurrencyTypes.USD: "USD_OIS_SOFR",
            CurrencyTypes.EUR: "EUR_OIS_ESTR",
        }

        if derivative._currency not in curve_name_map:
            raise LibError(f"No default OIS curve for currency {derivative._currency}")

        # Get discount curve (for discounting cashflows)
        discount_curve_name = curve_name_map[derivative._currency]
        discount_model = getattr(self.model.curves, discount_curve_name)

        # Get index curve (for forward rate projection)
        index_curve_name = derivative._floating_index.name
        index_model = getattr(self.model.curves, index_curve_name)

        # Build discount curve cache
        disc_curve_key = tuple(discount_model.swap_times)
        disc_cache = self._cached_curve(
            disc_curve_key,
            discount_model.swap_rates,
            discount_model.swap_times,
            discount_model.year_fracs,
            discount_model._interp_type
        )

        disc_times = disc_cache["times"]
        disc_dfs = disc_cache["dfs"]
        disc_jac = disc_cache["jac"]
        disc_hess = disc_cache["hess"]

        # Build index curve cache (may be same as discount curve)
        if index_curve_name == discount_curve_name:
            idx_times = disc_times
            idx_dfs = disc_dfs
            idx_jac = disc_jac
            idx_hess = disc_hess
        else:
            idx_curve_key = tuple(index_model.swap_times)
            idx_cache = self._cached_curve(
                idx_curve_key,
                index_model.swap_rates,
                index_model.swap_times,
                index_model.year_fracs,
                index_model._interp_type
            )
            idx_times = idx_cache["times"]
            idx_dfs = idx_cache["dfs"]
            idx_jac = idx_cache["jac"]
            idx_hess = idx_cache["hess"]

        # Extract FRN cashflow data
        dc_type = derivative._dc_type
        value_dt = discount_model._value_dt
        value_time = times_from_dates(value_dt, value_dt, dc_type)

        # Convert payment dates to times
        payment_times = jnp.array(
            [times_from_dates(dt, value_dt, dc_type) for dt in derivative._payment_dts]
        )

        # Convert start/end accrual dates to times
        start_times = jnp.array(
            [times_from_dates(dt, value_dt, dc_type) for dt in derivative._start_accrued_dts]
        )

        end_times = jnp.array(
            [times_from_dates(dt, value_dt, dc_type) for dt in derivative._end_accrued_dts]
        )

        # Year fractions for payment calculation
        pay_alphas = jnp.array(derivative._year_fracs)

        # Spreads (quoted margin)
        spreads = jnp.full_like(pay_alphas, derivative._quoted_margin)

        # Notionals (constant for FRN unless amortizing)
        notionals = jnp.full_like(pay_alphas, derivative._face_value)

        # Principal at maturity
        principal = derivative._face_value

        # FRNs are always from investor perspective (receive coupons + principal)
        leg_sign = +1.0

        # First fixing rate handling
        first_fixing_rate = derivative._first_fixing_rate if derivative._first_fixing_rate is not None else 0.0
        override_first = derivative._first_fixing_rate is not None

        # Determine if we need separate index curve
        use_separate_index = index_curve_name != discount_curve_name

        # Function to price floating leg + principal
        def pv_fn_combined(dfs):
            # Call _float_leg_jax with appropriate parameters
            if use_separate_index:
                float_pv = self._float_leg_jax(
                    dfs=dfs,
                    times=disc_times,
                    disc_interp_type=discount_model._interp_type,
                    idx_interp_type=index_model._interp_type,
                    payment_times=payment_times,
                    start_times=start_times,
                    end_times=end_times,
                    pay_alphas=pay_alphas,
                    spreads=spreads,
                    notionals=notionals,
                    principal=0.0,
                    leg_sign=leg_sign,
                    value_time=value_time,
                    first_fixing_rate=first_fixing_rate,
                    override_first=override_first,
                    idx_times=idx_times,
                    idx_dfs=idx_dfs
                )
            else:
                # Single curve case - index and discount are the same
                float_pv = self._float_leg_jax(
                    dfs=dfs,
                    times=disc_times,
                    disc_interp_type=discount_model._interp_type,
                    idx_interp_type=index_model._interp_type,
                    payment_times=payment_times,
                    start_times=start_times,
                    end_times=end_times,
                    pay_alphas=pay_alphas,
                    spreads=spreads,
                    notionals=notionals,
                    principal=0.0,
                    leg_sign=leg_sign,
                    value_time=value_time,
                    first_fixing_rate=first_fixing_rate,
                    override_first=override_first
                )

            # Add principal repayment at maturity
            maturity_time = times_from_dates(derivative._maturity_dt, value_dt, dc_type)
            if maturity_time > value_time:
                interp = InterpolatorAd(discount_model._interp_type)
                df_maturity = interp.simple_interpolate(
                    maturity_time,
                    disc_times,
                    dfs,
                    discount_model._interp_type.value
                )
                principal_pv = principal * leg_sign * df_maturity[0] if jnp.ndim(df_maturity) > 0 else principal * leg_sign * df_maturity
            else:
                principal_pv = 0.0

            return float_pv + principal_pv


        # Initialize results
        value = None
        delta = None
        gamma = None

        # Compute VALUE
        if RequestTypes.VALUE in reqs:
            val = pv_fn_combined(disc_dfs)
            val_scalar = float(jnp.atleast_1d(val).item() if jnp.ndim(val) == 0 else val.squeeze())
            value = Valuation(amount=val_scalar, currency=derivative._currency)

        # For DELTA and GAMMA, we need to handle dual-curve sensitivities
        # For now, implement single-curve case (discount curve sensitivities only)
        need_grad = RequestTypes.DELTA in reqs or RequestTypes.GAMMA in reqs

        if need_grad:
            # Check if discount and index curves are the same
            if index_curve_name == discount_curve_name:
                # Single curve case - easier
                # Compute gradient directly
                grad_dfs = grad(lambda d: jnp.squeeze(pv_fn_combined(d)))(disc_dfs)

                # Compute DELTA
                if RequestTypes.DELTA in reqs:
                    sensitivities = jnp.dot(grad_dfs, disc_jac)
                    sensies = [float(x) * 1e-4 for x in sensitivities]

                    curve_type_map = {
                        CurrencyTypes.GBP: CurveTypes.GBP_OIS_SONIA,
                        CurrencyTypes.USD: CurveTypes.USD_OIS_SOFR,
                        CurrencyTypes.EUR: CurveTypes.EUR_OIS_ESTR,
                    }
                    curve_type = curve_type_map.get(derivative._currency, CurveTypes.GBP_OIS_SONIA)

                    delta = Delta(
                        risk_ladder=sensies,
                        tenors=to_tenor(discount_model.swap_times),
                        currency=derivative._currency,
                        curve_type=curve_type,
                    )

                # Compute GAMMA
                if RequestTypes.GAMMA in reqs:
                    hess_dfs = hessian(lambda d: jnp.squeeze(pv_fn_combined(d)))(disc_dfs)

                    # Chain rule for second derivatives
                    term1 = disc_jac.T @ hess_dfs @ disc_jac
                    term2 = jnp.sum(grad_dfs[:, None, None] * disc_hess, axis=0)
                    gammas = term1 + term2

                    # Convert to bp^2
                    gammas = np.array(gammas, dtype=np.float64) * 1e-8

                    curve_type = curve_type_map.get(derivative._currency, CurveTypes.GBP_OIS_SONIA)

                    gamma = Gamma(
                        risk_ladder=gammas,
                        tenors=to_tenor(discount_model.swap_times),
                        currency=derivative._currency,
                        curve_type=curve_type,
                    )
            else:
                # Dual curve case - more complex (TODO: implement cross-curve sensitivities)
                raise LibError("Dual-curve FRN delta/gamma not yet implemented. "
                             "Use same curve for discounting and projection.")

        # Cashflows extraction
        cashflows = None
        if RequestTypes.CASHFLOWS in reqs:
            all_cashflows = []

            # Value the FRN to populate cashflow data
            derivative.value(self.model.value_dt, discount_model, index_model)

            # Extract floating coupon cashflows
            num_payments = len(derivative._payment_dts)

            for i in range(num_payments):
                payment_dt = derivative._payment_dts[i]
                coupon_amt = derivative._coupon_payments[i]

                # Extract principal component from last payment
                # (FRN value() adds principal to last payment's PV)
                is_last_payment = (i == num_payments - 1)

                # Coupon cashflow
                if abs(coupon_amt) > 1e-10:
                    coupon_fraction = derivative._rates[i]  # Floating rate + margin

                    cf_item = CashflowItem(
                        payment_date=payment_dt,
                        notional=derivative._face_value,
                        payment_fraction=coupon_fraction,
                        accrual_period=float(derivative._year_fracs[i]),
                        amount=float(coupon_amt),
                        discount_factor=float(derivative._payment_dfs[i]),
                        discounted_amount=float(coupon_amt * derivative._payment_dfs[i]),
                        leg_type="Floating_Coupon"
                    )
                    all_cashflows.append(cf_item)

                # Principal cashflow at maturity
                if is_last_payment:
                    principal_amt = derivative._face_value
                    df = derivative._payment_dfs[i] if i < len(derivative._payment_dfs) else 0.0

                    cf_item = CashflowItem(
                        payment_date=payment_dt,
                        notional=principal_amt,
                        payment_fraction=1.0,  # Principal repayment
                        accrual_period=0.0,  # No accrual for principal
                        amount=float(principal_amt),
                        discount_factor=float(df),
                        discounted_amount=float(principal_amt * df),
                        leg_type="Principal"
                    )
                    all_cashflows.append(cf_item)

            cashflows = Cashflows(all_cashflows, derivative._currency)

        return AnalyticsResult(value=value, risk=delta, gamma=gamma, cashflows=cashflows)

    def _compute_yoy_iis(self, derivative, reqs):
        """Compute analytics for Year-on-Year Inflation Swaps (VALUE, DELTA, GAMMA).

        Args:
            derivative: YoYInflationSwap instance
            reqs: Set of RequestTypes (VALUE, DELTA, GAMMA)

        Returns:
            AnalyticsResult with value, risk (delta), and gamma
        """
        # Get inflation curve from model based on index type and currency
        # Map from (currency, index_type) to curve attribute name
        inflation_curve_map = {
            (CurrencyTypes.GBP, "UK_RPI"): "GBP_RPI_INFLATION",
            (CurrencyTypes.GBP, "UK_CPI"): "GBP_CPI_INFLATION",
            (CurrencyTypes.USD, "US_CPI_U"): "USD_CPI_INFLATION",
            (CurrencyTypes.EUR, "EUR_HICP"): "EUR_HICP_INFLATION",
        }

        # Get discount curve based on currency
        discount_curve_map = {
            CurrencyTypes.GBP: "GBP_OIS_SONIA",
            CurrencyTypes.USD: "USD_OIS_SOFR",
            CurrencyTypes.EUR: "EUR_OIS_ESTR",
        }

        currency = derivative._inflation_index._currency
        index_type_name = derivative._inflation_index._index_type.name

        # Get discount curve
        if currency not in discount_curve_map:
            raise LibError(f"No default OIS curve for currency {currency}")

        discount_curve_name = discount_curve_map[currency]
        discount_curve = getattr(self.model.curves, discount_curve_name, None)

        if discount_curve is None:
            raise LibError(f"Discount curve {discount_curve_name} not found in model")

        # Get inflation curve
        inflation_curve_key = (currency, index_type_name)
        if inflation_curve_key not in inflation_curve_map:
            raise LibError(
                f"No inflation curve mapping for {currency.name} {index_type_name}. "
                f"Add to model.curves as {currency.name}_{index_type_name}_INFLATION"
            )

        inflation_curve_name = inflation_curve_map[inflation_curve_key]
        inflation_curve = getattr(self.model.curves, inflation_curve_name, None)

        if inflation_curve is None:
            raise LibError(f"Inflation curve {inflation_curve_name} not found in model")

        # Import JAX dependencies
        from jax import grad, hessian, jacrev, jacfwd
        import jax.numpy as jnp
        from cavour.utils.helpers import times_from_dates
        from cavour.market.curves.interpolator_ad import InterpolatorAd
        from functools import partial

        # Get cached discount curve data
        disc_curve_key = tuple(discount_curve.swap_times)
        disc_cache = self._cached_curve(
            disc_curve_key,
            discount_curve.swap_rates,
            discount_curve.swap_times,
            discount_curve.year_fracs,
            discount_curve._interp_type
        )
        disc_times = disc_cache["times"]
        disc_dfs = disc_cache["dfs"]
        disc_jac = disc_cache["jac"]
        disc_hess = disc_cache["hess"]

        # Get inflation curve data (use breakeven rates as inputs)
        # Note: _times includes t=0, _dfs includes 1.0 at t=0
        # IMPORTANT: Convert to plain numpy arrays FIRST, before any JAX transformations
        # This prevents JAX tracer leaks from contaminating the curve object
        try:
            # Try to convert directly (will work if no tracers)
            infl_times_np = np.array(inflation_curve._times, dtype=np.float64).copy()
            infl_factors_np = np.array(inflation_curve._dfs, dtype=np.float64).copy()
        except:
            # If curve is contaminated with tracers, rebuild from breakeven rates
            infl_breakeven_rates_temp = [zcis._fixed_rate for zcis in inflation_curve._used_swaps]
            infl_times_rebuild, infl_factors_rebuild = inflation_curve._build_curve_ad(
                jnp.array(infl_breakeven_rates_temp))
            infl_times_np = np.array(infl_times_rebuild, dtype=np.float64).copy()
            infl_factors_np = np.array(infl_factors_rebuild, dtype=np.float64).copy()

        # Now convert to JAX arrays for use in AD
        infl_times = jnp.array(infl_times_np)
        infl_factors = jnp.array(infl_factors_np)
        # Extract breakeven rates from ZCIS instruments
        infl_breakeven_rates = [zcis._fixed_rate for zcis in inflation_curve._used_swaps]

        # Prepare swap leg parameters
        dc_type = derivative._fixed_leg._dc_type
        value_time = times_from_dates(self.model.value_dt, self.model.value_dt, dc_type)

        # Fixed leg parameters
        fixed_payment_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                         for dt in derivative._fixed_leg._payment_dts])
        fixed_alphas = jnp.array(derivative._fixed_leg._year_fracs)
        fixed_coupons = jnp.full_like(fixed_alphas, derivative._fixed_leg._cpn)
        fixed_notionals = jnp.full_like(fixed_alphas, derivative._fixed_leg._notional)
        fixed_principal = derivative._fixed_leg._principal
        fixed_leg_sign = +1.0 if derivative._fixed_leg._leg_type == SwapTypes.RECEIVE else -1.0

        # YoY inflation leg parameters
        yoy_payment_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                       for dt in derivative._inflation_leg._payment_dts])
        yoy_start_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                     for dt in derivative._inflation_leg._yoy_start_dts])
        yoy_end_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                   for dt in derivative._inflation_leg._yoy_end_dts])
        yoy_alphas = jnp.array(derivative._inflation_leg._year_fracs)
        yoy_spread = derivative._inflation_leg._spread
        yoy_notionals = jnp.full_like(yoy_alphas, derivative._inflation_leg._notional)
        yoy_leg_sign = +1.0 if derivative._inflation_leg._leg_type == SwapTypes.RECEIVE else -1.0

        # Define JAX pricing function for YoY inflation leg
        def price_yoy_inflation_leg_jax(disc_dfs_var, infl_factors_var, disc_times_var, infl_times_var):
            """Price YoY inflation leg using JAX-compatible operations."""
            interp_disc = InterpolatorAd(discount_curve._interp_type)
            interp_infl = InterpolatorAd(inflation_curve._interp_type)

            # Discount factors for value date and payment dates
            df_val = jnp.atleast_1d(interp_disc.simple_interpolate(
                value_time, disc_times_var, disc_dfs_var, discount_curve._interp_type.value))
            df_pmts = jnp.atleast_1d(interp_disc.simple_interpolate(
                yoy_payment_times, disc_times_var, disc_dfs_var, discount_curve._interp_type.value))

            # Inflation factors at YoY start and end dates
            # Note: inflation_factors[0] = 1.0 at t=0, so factors grow with inflation
            infl_start = jnp.atleast_1d(interp_infl.simple_interpolate(
                yoy_start_times, infl_times_var, infl_factors_var, inflation_curve._interp_type.value))
            infl_end = jnp.atleast_1d(interp_infl.simple_interpolate(
                yoy_end_times, infl_times_var, infl_factors_var, inflation_curve._interp_type.value))

            # YoY inflation rates: (I_end / I_start) - 1
            yoy_rates = (infl_end / infl_start) - 1.0

            # Total rates including spread
            total_rates = yoy_rates + yoy_spread

            # Payments: notional × year_frac × (yoy_rate + spread)
            payments = yoy_notionals * yoy_alphas * total_rates

            # Mask for future cashflows
            mask = yoy_payment_times > value_time

            # Relative discount factors (df_val should be ~1.0 at value date)
            df_rel = df_pmts / jnp.squeeze(df_val)

            # PV of payments (only include future cashflows)
            pv_payments = jnp.where(mask, payments * df_rel, 0.0)

            # Sum payments
            leg_pv = jnp.sum(pv_payments)
            return yoy_leg_sign * leg_pv

        # Initialize results
        value = None
        delta = None
        gamma = None
        cashflows = None

        # Compute VALUE
        if RequestTypes.VALUE in reqs:
            # Fixed leg PV
            fixed_payments = fixed_coupons * fixed_alphas * fixed_notionals
            fixed_pv = self._price_fixed_leg_jax(
                dfs=disc_dfs,
                times=disc_times,
                interp_type=discount_curve._interp_type,
                payment_times=fixed_payment_times,
                payments=fixed_payments,
                principal=fixed_principal,
                notional=derivative._fixed_leg._notional,
                leg_sign=fixed_leg_sign,
                value_time=value_time
            )

            # YoY inflation leg PV
            yoy_pv = price_yoy_inflation_leg_jax(disc_dfs, infl_factors, disc_times, infl_times)

            # Total PV
            total_pv = float(jnp.squeeze(fixed_pv)) + float(jnp.squeeze(yoy_pv))
            value = Valuation(amount=total_pv, currency=currency)

        # Compute DELTA (multi-curve sensitivities)
        if RequestTypes.DELTA in reqs:
            # Define PV functions for each curve

            # 1. Fixed leg sensitivity to discount curve
            def pv_fixed_disc_fn(disc_dfs_var):
                fixed_payments = fixed_coupons * fixed_alphas * fixed_notionals
                return self._price_fixed_leg_jax(
                    dfs=disc_dfs_var, times=disc_times,
                    interp_type=discount_curve._interp_type,
                    payment_times=fixed_payment_times,
                    payments=fixed_payments, principal=fixed_principal,
                    notional=derivative._fixed_leg._notional,
                    leg_sign=fixed_leg_sign, value_time=value_time
                )

            # 2. YoY leg sensitivity to discount curve (holding inflation curve fixed)
            def pv_yoy_disc_fn(disc_dfs_var):
                return price_yoy_inflation_leg_jax(disc_dfs_var, infl_factors, disc_times, infl_times)

            # 3. YoY leg sensitivity to inflation curve (holding discount curve fixed)
            def pv_yoy_infl_fn(infl_factors_var):
                return price_yoy_inflation_leg_jax(disc_dfs, infl_factors_var, disc_times, infl_times)

            # Gradients w.r.t. discount factors
            grad_fixed_disc = grad(lambda d: jnp.squeeze(pv_fixed_disc_fn(d)))(disc_dfs)
            grad_yoy_disc = grad(lambda d: jnp.squeeze(pv_yoy_disc_fn(d)))(disc_dfs)
            grad_total_disc = grad_fixed_disc + grad_yoy_disc

            # Chain rule: sensitivity to discount curve rates
            disc_sensitivities = jnp.dot(grad_total_disc, disc_jac)
            disc_sensies = [float(x) * 1e-4 for x in disc_sensitivities]  # Convert to bp

            # Gradient w.r.t. inflation factors
            grad_yoy_infl = grad(lambda i: jnp.squeeze(pv_yoy_infl_fn(i)))(infl_factors)

            # For inflation curve, we need Jacobian of factors w.r.t. breakeven rates
            # Use _build_curve_ad to get this Jacobian
            def inflation_factors_from_rates(rates):
                _, factors_ad = inflation_curve._build_curve_ad(rates)
                return factors_ad

            infl_breakeven_rates_jax = jnp.array(infl_breakeven_rates)
            infl_jac = jacrev(inflation_factors_from_rates)(infl_breakeven_rates_jax)

            # Chain rule: sensitivity to inflation breakeven rates
            infl_sensitivities = jnp.dot(grad_yoy_infl, infl_jac)
            infl_sensies = [float(x) * 1e-4 for x in infl_sensitivities]  # Convert to bp

            # Create multi-curve Delta object using Risk container
            from cavour.requests.results import Risk, Delta

            disc_curve_type_map = {
                CurrencyTypes.GBP: CurveTypes.GBP_OIS_SONIA,
                CurrencyTypes.USD: CurveTypes.USD_OIS_SOFR,
                CurrencyTypes.EUR: CurveTypes.EUR_OIS_ESTR,
            }
            disc_curve_type = disc_curve_type_map.get(currency, CurveTypes.GBP_OIS_SONIA)

            infl_curve_type_map = {
                (CurrencyTypes.GBP, "UK_RPI"): CurveTypes.GBP_RPI_INFLATION,
                (CurrencyTypes.GBP, "UK_CPI"): CurveTypes.GBP_CPI_INFLATION,
                (CurrencyTypes.USD, "US_CPI_U"): CurveTypes.USD_CPI_INFLATION,
                (CurrencyTypes.EUR, "EUR_HICP"): CurveTypes.EUR_HICP_INFLATION,
            }
            infl_curve_type = infl_curve_type_map.get(
                (currency, index_type_name), CurveTypes.GBP_RPI_INFLATION)

            # Create Delta objects for each curve
            disc_delta_obj = Delta(
                risk_ladder=disc_sensies,
                tenors=to_tenor(discount_curve.swap_times),
                currency=currency,
                curve_type=disc_curve_type
            )

            infl_delta_obj = Delta(
                risk_ladder=infl_sensies,
                tenors=to_tenor(inflation_curve.swap_times),
                currency=currency,
                curve_type=infl_curve_type
            )

            delta = Risk([disc_delta_obj, infl_delta_obj])

        # Compute GAMMA (multi-curve second-order sensitivities)
        if RequestTypes.GAMMA in reqs:
            # Define PV functions (same as DELTA)
            def pv_fixed_disc_fn(disc_dfs_var):
                fixed_payments = fixed_coupons * fixed_alphas * fixed_notionals
                return self._price_fixed_leg_jax(
                    dfs=disc_dfs_var, times=disc_times,
                    interp_type=discount_curve._interp_type,
                    payment_times=fixed_payment_times,
                    payments=fixed_payments, principal=fixed_principal,
                    notional=derivative._fixed_leg._notional,
                    leg_sign=fixed_leg_sign, value_time=value_time
                )

            def pv_yoy_disc_fn(disc_dfs_var):
                return price_yoy_inflation_leg_jax(disc_dfs_var, infl_factors, disc_times, infl_times)

            def pv_yoy_infl_fn(infl_factors_var):
                return price_yoy_inflation_leg_jax(disc_dfs, infl_factors_var, disc_times, infl_times)

            # Total PV function for discount curve
            def pv_total_disc_fn(disc_dfs_var):
                return jnp.squeeze(pv_fixed_disc_fn(disc_dfs_var)) + jnp.squeeze(pv_yoy_disc_fn(disc_dfs_var))

            # Gradients (needed for chain rule)
            grad_total_disc = grad(pv_total_disc_fn)(disc_dfs)
            grad_yoy_infl = grad(lambda i: jnp.squeeze(pv_yoy_infl_fn(i)))(infl_factors)

            # Hessians w.r.t. discount factors
            hess_total_disc = hessian(pv_total_disc_fn)(disc_dfs)

            # Hessian w.r.t. inflation factors
            hess_yoy_infl = hessian(lambda i: jnp.squeeze(pv_yoy_infl_fn(i)))(infl_factors)

            # Get inflation curve Jacobian and Hessian
            def inflation_factors_from_rates(rates):
                _, factors_ad = inflation_curve._build_curve_ad(rates)
                return factors_ad

            infl_breakeven_rates_jax = jnp.array(infl_breakeven_rates)
            infl_jac = jacrev(inflation_factors_from_rates)(infl_breakeven_rates_jax)
            infl_hess = jacfwd(jacrev(inflation_factors_from_rates))(infl_breakeven_rates_jax)

            # Chain rule for discount curve gamma
            disc_gamma_term1 = disc_jac.T @ hess_total_disc @ disc_jac
            disc_gamma_term2 = jnp.sum(grad_total_disc[:, None, None] * disc_hess, axis=0)
            disc_gamma = disc_gamma_term1 + disc_gamma_term2
            disc_gamma = np.array(disc_gamma, dtype=np.float64) * 1e-8  # Convert to bp²

            # Chain rule for inflation curve gamma
            infl_gamma_term1 = infl_jac.T @ hess_yoy_infl @ infl_jac
            infl_gamma_term2 = jnp.sum(grad_yoy_infl[:, None, None] * infl_hess, axis=0)
            infl_gamma = infl_gamma_term1 + infl_gamma_term2
            infl_gamma = np.array(infl_gamma, dtype=np.float64) * 1e-8  # Convert to bp²

            # TODO: Cross-curve gamma (discount × inflation) - currently zero
            # This would require computing d²PV/(d_disc × d_infl)

            # Create multi-curve Gamma object using Risk container
            from cavour.requests.results import Risk, Gamma

            disc_curve_type_map = {
                CurrencyTypes.GBP: CurveTypes.GBP_OIS_SONIA,
                CurrencyTypes.USD: CurveTypes.USD_OIS_SOFR,
                CurrencyTypes.EUR: CurveTypes.EUR_OIS_ESTR,
            }
            disc_curve_type = disc_curve_type_map.get(currency, CurveTypes.GBP_OIS_SONIA)

            infl_curve_type_map = {
                (CurrencyTypes.GBP, "UK_RPI"): CurveTypes.GBP_RPI_INFLATION,
                (CurrencyTypes.GBP, "UK_CPI"): CurveTypes.GBP_CPI_INFLATION,
                (CurrencyTypes.USD, "US_CPI_U"): CurveTypes.USD_CPI_INFLATION,
                (CurrencyTypes.EUR, "EUR_HICP"): CurveTypes.EUR_HICP_INFLATION,
            }
            infl_curve_type = infl_curve_type_map.get(
                (currency, index_type_name), CurveTypes.GBP_RPI_INFLATION)

            disc_gamma_obj = Gamma(
                risk_ladder=disc_gamma,
                tenors=to_tenor(discount_curve.swap_times),
                currency=currency,
                curve_type=disc_curve_type
            )

            infl_gamma_obj = Gamma(
                risk_ladder=infl_gamma,
                tenors=to_tenor(inflation_curve.swap_times),
                currency=currency,
                curve_type=infl_curve_type
            )

            gamma = Risk([disc_gamma_obj, infl_gamma_obj])

        # CASHFLOWS extraction
        if RequestTypes.CASHFLOWS in reqs:
            all_cashflows = []

            # Value the swap to populate cashflow data
            derivative.value(self.model.value_dt, discount_curve, inflation_curve)

            # Extract fixed leg cashflows
            fixed_leg_type = "Fixed_Pay" if derivative._fixed_leg_type == SwapTypes.PAY else "Fixed_Rec"
            fixed_cashflows = self._extract_leg_cashflows(derivative._fixed_leg, fixed_leg_type)
            all_cashflows.extend(fixed_cashflows)

            # Extract YoY inflation leg cashflows
            yoy_leg_type = "YoY_Inflation_Rec" if derivative._fixed_leg_type == SwapTypes.PAY else "YoY_Inflation_Pay"

            # Check if YoY leg has been valued (has payment_pvs attribute)
            if hasattr(derivative._inflation_leg, '_payment_pvs') and derivative._inflation_leg._payment_pvs:
                sign = +1.0 if 'Rec' in yoy_leg_type else -1.0

                for i in range(len(derivative._inflation_leg._payment_dts)):
                    notional = float(derivative._inflation_leg._notional)

                    # YoY inflation leg payment = notional × year_frac × (yoy_rate + spread)
                    if hasattr(derivative._inflation_leg, '_yoy_rates') and i < len(derivative._inflation_leg._yoy_rates):
                        yoy_rate = float(derivative._inflation_leg._yoy_rates[i])
                        spread = float(derivative._inflation_leg._spread)
                        total_rate = yoy_rate + spread
                    else:
                        # Compute from payment amount
                        year_frac = float(derivative._inflation_leg._year_fracs[i])
                        if notional != 0 and year_frac != 0:
                            total_rate = float(derivative._inflation_leg._payments[i]) / (notional * year_frac)
                        else:
                            total_rate = 0.0

                    payment_amt = float(derivative._inflation_leg._payments[i])
                    signed_amt = sign * payment_amt
                    signed_pv = sign * float(derivative._inflation_leg._payment_pvs[i])

                    cf_item = CashflowItem(
                        payment_date=derivative._inflation_leg._payment_dts[i],
                        notional=notional,
                        payment_fraction=total_rate,
                        accrual_period=float(derivative._inflation_leg._year_fracs[i]),
                        amount=signed_amt,
                        discount_factor=float(derivative._inflation_leg._payment_dfs[i]),
                        discounted_amount=signed_pv,
                        leg_type=yoy_leg_type
                    )
                    all_cashflows.append(cf_item)

            cashflows = Cashflows(all_cashflows, currency)

        return AnalyticsResult(value=value, risk=delta, gamma=gamma, cashflows=cashflows)


    def _compute_xccy(self, derivative, reqs, collateral_type=None):
        """Compute analytics for cross-currency swaps (VALUE, DELTA, GAMMA).

        Handles XccyFixFloat, XccyBasisSwap, and XccyFixFix swaps.
        Uses JAX automatic differentiation for multi-curve sensitivities.

        Args:
            derivative: XCCY swap instance (XccyFixFloat, XccyBasisSwap, or XccyFixFix)
            reqs: Set of RequestTypes (VALUE, DELTA, GAMMA)

        Returns:
            AnalyticsResult with value, risk (multi-curve Delta), and gamma
        """
        from jax import grad
        import jax.numpy as jnp
        from cavour.utils.helpers import times_from_dates

        # Get curves from model
        domestic_model = getattr(self.model.curves, derivative._domestic_floating_index.name)
        foreign_model = getattr(self.model.curves, derivative._foreign_floating_index.name)

        # Get XCCY curve and spot FX
        foreign_code = derivative._foreign_currency.name
        domestic_code = derivative._domestic_currency.name
        xccy_curve_name = f"{foreign_code}_{domestic_code}_BASIS"

        try:
            xccy_curve = getattr(self.model.curves, xccy_curve_name)
            spot_fx = xccy_curve._spot_fx
        except AttributeError:
            raise LibError(f"XCCY curve {xccy_curve_name} not found in model. "
                         f"Available curves: {[attr for attr in dir(self.model.curves) if not attr.startswith('_')]}")

        # Build domestic OIS curve arrays
        dom_curve_key = tuple(domestic_model.swap_times)
        dom_cache = self._cached_curve(
            dom_curve_key,
            domestic_model.swap_rates,
            domestic_model.swap_times,
            domestic_model.year_fracs,
            domestic_model._interp_type
        )
        dom_times = dom_cache["times"]
        dom_dfs = dom_cache["dfs"]

        # Build foreign OIS curve arrays
        for_curve_key = tuple(foreign_model.swap_times)
        for_cache = self._cached_curve(
            for_curve_key,
            foreign_model.swap_rates,
            foreign_model.swap_times,
            foreign_model.year_fracs,
            foreign_model._interp_type
        )
        for_times = for_cache["times"]
        for_dfs = for_cache["dfs"]

        # Get XCCY curve arrays
        # Note: XCCY curve times are in ACT_365F, but we'll use them to interpolate
        # foreign leg payment times in ACT_360. This creates a small time mismatch
        # (similar to how .value() method handles it)
        xccy_times = jnp.array(xccy_curve._times)
        xccy_dfs = jnp.array(xccy_curve._dfs)

        # Prepare leg parameters for JAX computation
        dc_type = derivative._domestic_leg._dc_type
        value_time = times_from_dates(self.model.value_dt, self.model.value_dt, dc_type)

        # Domestic leg parameters
        dom_payment_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                       for dt in derivative._domestic_leg._payment_dts])
        dom_start_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                     for dt in derivative._domestic_leg._start_accrued_dts])
        dom_end_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                   for dt in derivative._domestic_leg._end_accrued_dts])
        dom_alphas = jnp.array(derivative._domestic_leg._year_fracs)
        dom_spreads = jnp.full_like(dom_alphas, derivative._domestic_leg._spread)
        dom_notionals = jnp.array(derivative._domestic_leg._notional_array or
                                  [derivative._domestic_leg._notional] * len(dom_alphas))
        dom_principal = derivative._domestic_leg._principal
        dom_leg_sign = +1.0 if derivative._domestic_leg._leg_type == SwapTypes.RECEIVE else -1.0

        # Foreign leg parameters
        # For forward rates: use foreign leg's day count (ACT_360) to match foreign OIS curve
        # For discounting: use XCCY curve's day count (ACT_365F) for proper interpolation
        for_dc_type = derivative._foreign_leg._dc_type  # ACT_360 for forward rates
        xccy_dc_type = xccy_curve._dc_type  # ACT_365F for discounting

        # Payment times for discounting (must match XCCY curve times)
        for_payment_times = jnp.array([times_from_dates(dt, self.model.value_dt, xccy_dc_type)
                                       for dt in derivative._foreign_leg._payment_dts])
        # Start/end times for forward rates (must match foreign OIS curve times)
        for_start_times = jnp.array([times_from_dates(dt, self.model.value_dt, for_dc_type)
                                     for dt in derivative._foreign_leg._start_accrued_dts])
        for_end_times = jnp.array([times_from_dates(dt, self.model.value_dt, for_dc_type)
                                   for dt in derivative._foreign_leg._end_accrued_dts])
        for_alphas = jnp.array(derivative._foreign_leg._year_fracs)
        for_spreads = jnp.full_like(for_alphas, derivative._foreign_leg._spread)
        for_notionals = jnp.array(derivative._foreign_leg._notional_array or
                                  [derivative._foreign_leg._notional] * len(for_alphas))
        for_principal = derivative._foreign_leg._principal
        for_leg_sign = +1.0 if derivative._foreign_leg._leg_type == SwapTypes.RECEIVE else -1.0

        # Compute effective and maturity times for notional exchanges
        # Use discount curve's day count for discounting times
        dom_effective_time = times_from_dates(derivative._effective_dt, self.model.value_dt, dc_type)
        dom_maturity_time = times_from_dates(derivative._maturity_dt, self.model.value_dt, dc_type)
        # Foreign leg: use XCCY curve's day count (ACT_365F) for discounting
        for_effective_time = times_from_dates(derivative._effective_dt, self.model.value_dt, xccy_dc_type)
        for_maturity_time = times_from_dates(derivative._maturity_dt, self.model.value_dt, xccy_dc_type)

        # Compute VALUE using JAX
        value = None
        if RequestTypes.VALUE in reqs:
            # Domestic leg PV (single curve)
            dom_pv = self._float_leg_jax(
                dfs=dom_dfs,
                times=dom_times,
                disc_interp_type=domestic_model._interp_type,
                idx_interp_type=domestic_model._interp_type,
                payment_times=dom_payment_times,
                start_times=dom_start_times,
                end_times=dom_end_times,
                pay_alphas=dom_alphas,
                spreads=dom_spreads,
                notionals=dom_notionals,
                principal=dom_principal,
                leg_sign=dom_leg_sign,
                value_time=value_time,
                first_fixing_rate=0.0,
                override_first=False,
                idx_times=None,
                idx_dfs=None,
                notional_exchange=derivative._domestic_leg._notional_exchange,
                notional_exchange_amount=derivative._domestic_leg._notional,
                effective_time=dom_effective_time,
                maturity_time=dom_maturity_time
            )

            # Foreign leg PV (dual curve: XCCY for discount, foreign OIS for index)
            for_pv = self._float_leg_jax(
                dfs=xccy_dfs,  # XCCY curve for discounting
                times=xccy_times,
                disc_interp_type=xccy_curve._interp_type,
                idx_interp_type=foreign_model._interp_type,
                payment_times=for_payment_times,
                start_times=for_start_times,
                end_times=for_end_times,
                pay_alphas=for_alphas,
                spreads=for_spreads,
                notionals=for_notionals,
                principal=for_principal,
                leg_sign=for_leg_sign,
                value_time=value_time,
                first_fixing_rate=0.0,
                override_first=False,
                idx_times=for_times,  # Foreign OIS for forward rates
                idx_dfs=for_dfs,
                notional_exchange=derivative._foreign_leg._notional_exchange,
                notional_exchange_amount=derivative._foreign_leg._notional,
                effective_time=for_effective_time,
                maturity_time=for_maturity_time
            )

            # Convert to scalars and compute total PV
            dom_pv_scalar = float(jnp.squeeze(dom_pv))
            for_pv_scalar = float(jnp.squeeze(for_pv))
            total_pv = dom_pv_scalar + for_pv_scalar / spot_fx
            value = Valuation(amount=total_pv, currency=derivative._domestic_currency)

        # Define PV functions for gradient/hessian computation (used by both DELTA and GAMMA)
        # These functions compute PV as a function of different curve variables

        # Domestic leg PV as function of domestic DFs
        def pv_dom_fn(dom_dfs_var):
            return self._float_leg_jax(
                dfs=dom_dfs_var, times=dom_times,
                disc_interp_type=domestic_model._interp_type,
                idx_interp_type=domestic_model._interp_type,
                payment_times=dom_payment_times,
                start_times=dom_start_times, end_times=dom_end_times,
                pay_alphas=dom_alphas, spreads=dom_spreads,
                notionals=dom_notionals, principal=dom_principal,
                leg_sign=dom_leg_sign, value_time=value_time,
                first_fixing_rate=0.0, override_first=False,
                notional_exchange=derivative._domestic_leg._notional_exchange,
                notional_exchange_amount=derivative._domestic_leg._notional,
                effective_time=dom_effective_time,
                maturity_time=dom_maturity_time
            )

        # Foreign leg PV as function of foreign OIS DFs (for direct forward rate effect)
        def pv_for_fn(for_ois_dfs_var):
            return self._float_leg_jax(
                dfs=xccy_dfs, times=xccy_times,  # XCCY curve for discounting (FIXED)
                disc_interp_type=xccy_curve._interp_type,
                idx_interp_type=foreign_model._interp_type,
                payment_times=for_payment_times,
                start_times=for_start_times, end_times=for_end_times,
                pay_alphas=for_alphas, spreads=for_spreads,
                notionals=for_notionals, principal=for_principal,
                leg_sign=for_leg_sign, value_time=value_time,
                first_fixing_rate=0.0, override_first=False,
                idx_times=for_times, idx_dfs=for_ois_dfs_var,  # Foreign OIS DFs (VARIABLE)
                notional_exchange=derivative._foreign_leg._notional_exchange,
                notional_exchange_amount=derivative._foreign_leg._notional,
                effective_time=for_effective_time,
                maturity_time=for_maturity_time
            )

        # Foreign leg PV as function of XCCY DFs (for basis delta and cross-gamma)
        def pv_xccy_fn(xccy_dfs_var):
            return self._float_leg_jax(
                dfs=xccy_dfs_var, times=xccy_times,  # XCCY curve for discounting (VARIABLE)
                disc_interp_type=xccy_curve._interp_type,
                idx_interp_type=foreign_model._interp_type,
                payment_times=for_payment_times,
                start_times=for_start_times, end_times=for_end_times,
                pay_alphas=for_alphas, spreads=for_spreads,
                notionals=for_notionals, principal=for_principal,
                leg_sign=for_leg_sign, value_time=value_time,
                first_fixing_rate=0.0, override_first=False,
                idx_times=for_times, idx_dfs=for_dfs,  # Foreign OIS DFs (FIXED)
                notional_exchange=derivative._foreign_leg._notional_exchange,
                notional_exchange_amount=derivative._foreign_leg._notional,
                effective_time=for_effective_time,
                maturity_time=for_maturity_time
            )

        # Wrapper functions for "original" DFs (excluding prepended t≈0)
        # IMPORTANT: DF(t≈0) = 1.0 is a boundary condition, NOT a curve parameter.
        # We prepended it to the curve grid for interpolation, but should NOT compute gradients w.r.t. it.

        def pv_dom_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs])
            return pv_dom_fn(full_dfs)

        def pv_for_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs])
            return pv_for_fn(full_dfs)

        def pv_xccy_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs])
            return pv_xccy_fn(full_dfs)

        # Compute DELTA using automatic differentiation
        delta = None
        if RequestTypes.DELTA in reqs:
            from cavour.utils.helpers import to_tenor

            # Extract original DFs (excluding prepended t≈0 if present)
            dom_dfs_original = dom_dfs[1:] if dom_times[0] < 1e-6 else dom_dfs

            # Compute gradients w.r.t. ORIGINAL DFs only (excluding DF(0)=1.0)
            grad_dom_dfs_original = grad(lambda d: jnp.squeeze(pv_dom_original_dfs(d)))(dom_dfs_original)

            # Chain rule: sensitivities to rates
            # Domestic OIS: simple chain rule
            dom_cache = self._cached_curve(
                tuple(domestic_model.swap_times),
                domestic_model.swap_rates,
                domestic_model.swap_times,
                domestic_model.year_fracs,
                domestic_model._interp_type
            )

            # The Jacobian has shape (n_dfs, n_rates) where n_dfs includes prepended point
            # Skip the first row (which is zeros for the prepended DF(0)=1.0)
            jac_dom_original = dom_cache["jac"][1:, :] if dom_times[0] < 1e-6 else dom_cache["jac"]
            delta_dom_rates = jnp.dot(grad_dom_dfs_original, jac_dom_original)
            delta_dom_rates = [float(x) * 1e-4 for x in delta_dom_rates]

            # Foreign OIS: Extract DFs and compute gradients/Jacobians
            for_ois_dfs_original = for_dfs[1:] if for_times[0] < 1e-6 else for_dfs
            grad_for_dfs_original = grad(lambda d: jnp.squeeze(pv_for_original_dfs(d)))(for_ois_dfs_original)
            jac_for_original = for_cache["jac"][1:, :] if for_times[0] < 1e-6 else for_cache["jac"]

            # XCCY: Extract DFs and compute gradients (needed for basis delta/gamma and cross-gamma)
            xccy_dfs_original = xccy_dfs[1:] if xccy_times[0] < 1e-6 else xccy_dfs
            grad_xccy_dfs_original = grad(lambda d: jnp.squeeze(pv_xccy_original_dfs(d)))(xccy_dfs_original)

            # Foreign OIS DELTA: Only direct effect on forward rates
            # IMPORTANT: XCCY curve is treated as FIXED market data when bumping foreign OIS rates.
            # In a risk scenario, we bump foreign OIS rates but XCCY basis spreads remain unchanged
            # (they are market observables calibrated at a point in time). Therefore, XCCY DFs do NOT
            # change when we bump foreign OIS, and we only include the direct effect on forward rates.
            #
            # This is different from a "market scenario" where changing foreign OIS would cause market
            # basis spreads to adjust, leading to XCCY re-calibration. For risk purposes, we compute
            # delta holding XCCY curve fixed, so only forward rate sensitivity matters.
            term1_foreign = jnp.dot(grad_for_dfs_original, jac_for_original)
            delta_for_rates_raw = term1_foreign

            # XCCY Basis: DELTA computation for basis spread curve
            # Compute sensitivity to basis spread changes (keeping both OIS curves fixed)
            # Only the foreign leg has sensitivity to XCCY curve (used for discounting)
            # Note: pv_xccy_fn, pv_xccy_original_dfs, and grad_xccy_dfs_original already computed above

            # Convert to GBP per bp
            # Foreign leg PV is in USD, divide by spot_fx to convert USD to GBP (spot_fx is USD/GBP)
            # Rates are stored in DECIMAL (0.052 for 5.2%), Jacobian is d(DFs)/d(rate_decimal)
            # 1bp = 0.0001 in decimal units → multiply by 1e-4
            delta_for_rates = [float(x) * 1e-4 / spot_fx for x in delta_for_rates_raw]

            # Chain rule: sensitivities to basis spreads
            # The XCCY curve has a Jacobian d(DFs)/d(basis_spreads) stored as _jac_basis
            # Check if JAX-based bootstrap was used (has _jac_basis attribute)
            if hasattr(xccy_curve, '_jac_basis') and xccy_curve._jac_basis is not None:
                # Get basis spread tenors from XCCY curve (needed for both DELTA and GAMMA)
                # Convert swap times to tenor strings
                basis_swap_tenors = to_tenor(xccy_curve.swap_times)

                # The Jacobian is already at pillar-level (one column per swap/pillar)
                jac_xccy_pillar = xccy_curve._jac_basis[1:, :] if xccy_times[0] < 1e-6 else xccy_curve._jac_basis

                # Compute delta: grad(PV, DFs) · Jacobian(DFs, pillar_spreads)
                delta_basis_rates_raw = jnp.dot(grad_xccy_dfs_original, jac_xccy_pillar)

                # Convert to GBP per bp
                # Foreign leg PV is in USD, divide by spot_fx to convert USD to GBP (spot_fx is USD/GBP)
                # Basis spreads are stored in DECIMAL (0.0030 for 30bp), Jacobian is d(DFs)/d(spread_decimal)
                # 1bp = 0.0001 in decimal units → multiply by 1e-4
                delta_basis_rates = [float(x) * 1e-4 / spot_fx for x in delta_basis_rates_raw]

                delta_basis = Delta(
                    risk_ladder=delta_basis_rates,
                    tenors=basis_swap_tenors,
                    currency=derivative._domestic_currency,
                    curve_type=CurveTypes.USD_GBP_BASIS,
                )
            else:
                # Fallback: no XCCY basis delta if Jacobian not available
                delta_basis = None

            # Create Delta objects for each curve
            delta_domestic = Delta(
                risk_ladder=delta_dom_rates,
                tenors=to_tenor(domestic_model.swap_times),
                currency=derivative._domestic_currency,
                curve_type=derivative._domestic_floating_index,
            )

            delta_foreign = Delta(
                risk_ladder=delta_for_rates,
                tenors=to_tenor(foreign_model.swap_times),
                currency=derivative._domestic_currency,
                curve_type=derivative._foreign_floating_index,
            )

            # Package deltas into Risk object
            # Include XCCY basis delta if available
            if delta_basis is not None:
                delta = Risk([delta_domestic, delta_foreign, delta_basis])
            else:
                delta = Risk([delta_domestic, delta_foreign])

        # Compute GAMMA using automatic differentiation
        gamma = None
        if RequestTypes.GAMMA in reqs:
            from cavour.utils.helpers import to_tenor

            # Extract original DFs (excluding prepended t≈0 if present) for all curves
            dom_dfs_original = dom_dfs[1:] if dom_times[0] < 1e-6 else dom_dfs
            for_ois_dfs_original = for_dfs[1:] if for_times[0] < 1e-6 else for_dfs
            xccy_dfs_original = xccy_dfs[1:] if xccy_times[0] < 1e-6 else xccy_dfs

            # Compute gradients and Jacobians (needed for gamma computation)
            grad_dom_dfs_original = grad(lambda d: jnp.squeeze(pv_dom_original_dfs(d)))(dom_dfs_original)
            jac_dom_original = dom_cache["jac"][1:, :] if dom_times[0] < 1e-6 else dom_cache["jac"]

            grad_for_dfs_original = grad(lambda d: jnp.squeeze(pv_for_original_dfs(d)))(for_ois_dfs_original)
            jac_for_original = for_cache["jac"][1:, :] if for_times[0] < 1e-6 else for_cache["jac"]

            grad_xccy_dfs_original = grad(lambda d: jnp.squeeze(pv_xccy_original_dfs(d)))(xccy_dfs_original)

            # Domestic OIS GAMMA
            # Compute Hessian w.r.t. domestic DFs
            hess_dom_dfs_original = hessian(lambda d: jnp.squeeze(pv_dom_original_dfs(d)))(dom_dfs_original)

            # Retrieve Hessian of curve bootstrapping (d²DFs/d(rates)²)
            hess_dom_curve = dom_cache["hess"][1:, :, :] if dom_times[0] < 1e-6 else dom_cache["hess"]

            # Chain rule for gamma: d²PV/d(rates)² = jac^T @ hess_dfs @ jac + sum(grad_dfs * hess_curve)
            # term1: main chain rule (treating curve as fixed mapping)
            # term2: correction for curve Hessian (derivative of Jacobian itself)
            term1_dom = jac_dom_original.T @ hess_dom_dfs_original @ jac_dom_original
            term2_dom = jnp.sum(grad_dom_dfs_original[:, None, None] * hess_dom_curve, axis=0)
            gammas_dom_matrix = term1_dom + term2_dom

            # Return FULL gamma matrix (not just diagonal)
            # Shape: (n_dom_rates, n_dom_rates)
            gammas_dom = gammas_dom_matrix

            # Convert to GBP per bp²
            # 1bp = 0.0001 in decimal → (1bp)² = 1e-8
            gammas_dom = np.array(gammas_dom, dtype=np.float64) * 1e-8

            # Foreign OIS GAMMA
            # Compute Hessian w.r.t. foreign DFs (direct effect through forward rates)
            hess_for_dfs_original = hessian(lambda d: jnp.squeeze(pv_for_original_dfs(d)))(for_ois_dfs_original)

            # Retrieve Hessian of curve bootstrapping
            hess_for_curve = for_cache["hess"][1:, :, :] if for_times[0] < 1e-6 else for_cache["hess"]

            # Chain rule for gamma - DIRECT effect (foreign OIS -> forward rates -> PV)
            term1_for = jac_for_original.T @ hess_for_dfs_original @ jac_for_original
            term2_for = jnp.sum(grad_for_dfs_original[:, None, None] * hess_for_curve, axis=0)
            gammas_for_matrix_direct = term1_for + term2_for

            # Foreign OIS GAMMA: Only direct effect on forward rates
            # IMPORTANT: XCCY curve is treated as FIXED when bumping foreign OIS rates.
            # Same rationale as for DELTA - we hold XCCY basis spreads fixed, so XCCY DFs
            # do not change. Therefore, only the direct effect on forward rates matters.
            gammas_for_matrix = gammas_for_matrix_direct

            # Return FULL gamma matrix (not just diagonal)
            # Shape: (n_for_rates, n_for_rates)
            gammas_for = gammas_for_matrix

            # Convert to GBP per bp²
            # Foreign leg PV is in USD, multiply by spot_fx to convert to GBP
            gammas_for = np.array(gammas_for, dtype=np.float64) * 1e-8 / spot_fx

            # Create Gamma objects for each curve
            gamma_domestic = Gamma(
                risk_ladder=gammas_dom,
                tenors=to_tenor(domestic_model.swap_times),
                currency=derivative._domestic_currency,
                curve_type=derivative._domestic_floating_index,
            )

            gamma_foreign = Gamma(
                risk_ladder=gammas_for,
                tenors=to_tenor(foreign_model.swap_times),
                currency=derivative._domestic_currency,
                curve_type=derivative._foreign_floating_index,
            )

            # XCCY Basis GAMMA
            # Compute Hessian w.r.t. XCCY DFs (if Jacobian available)
            if hasattr(xccy_curve, '_jac_basis') and xccy_curve._jac_basis is not None:
                # Get basis spread tenors from XCCY curve
                basis_swap_tenors = to_tenor(xccy_curve.swap_times)
                # Compute Hessian w.r.t. XCCY DFs
                hess_xccy_dfs_original = hessian(lambda d: jnp.squeeze(pv_xccy_original_dfs(d)))(xccy_dfs_original)

                # The Jacobian is already at pillar-level (one column per swap/pillar)
                jac_xccy_pillar = xccy_curve._jac_basis[1:, :] if xccy_times[0] < 1e-6 else xccy_curve._jac_basis

                # Chain rule for gamma (COMPLETE version with both terms)
                # term1: main chain rule (treating curve as fixed mapping)
                # term2: correction for curve Hessian (derivative of Jacobian itself)
                term1_xccy = jac_xccy_pillar.T @ hess_xccy_dfs_original @ jac_xccy_pillar

                # Check if curve Hessian is available (added in xccy_curve.py)
                if hasattr(xccy_curve, "_hess_basis") and xccy_curve._hess_basis is not None:
                    hess_xccy_curve = xccy_curve._hess_basis[1:, :, :] if xccy_times[0] < 1e-6 else xccy_curve._hess_basis
                    term2_xccy = jnp.sum(grad_xccy_dfs_original[:, None, None] * hess_xccy_curve, axis=0)
                    gammas_xccy_matrix = term1_xccy + term2_xccy
                else:
                    # Fallback to term1 only (will likely give zero or near-zero)
                    gammas_xccy_matrix = term1_xccy

                # Return FULL gamma matrix (not just diagonal)
                # Shape: (n_basis, n_basis)
                gammas_xccy = gammas_xccy_matrix

                # Convert to GBP per bp²
                # Foreign leg PV is in USD, divide by spot_fx to convert USD to GBP (spot_fx is USD/GBP)
                gammas_xccy = np.array(gammas_xccy, dtype=np.float64) * 1e-8 / spot_fx

                gamma_basis = Gamma(
                    risk_ladder=gammas_xccy,
                    tenors=basis_swap_tenors,
                    currency=derivative._domestic_currency,
                    curve_type=CurveTypes.USD_GBP_BASIS,
                )
            else:
                gamma_basis = None

            # Cross-Gamma: Foreign OIS <-> XCCY Basis
            # This captures how XCCY basis delta changes when foreign OIS rates move
            cross_gamma_for_basis = None
            if hasattr(xccy_curve, '_mixed_hess_foreign_basis') and xccy_curve._mixed_hess_foreign_basis is not None:
                # DEBUG: Print dimensions
                print(f"DEBUG Cross-gamma dimensions:")
                print(f"  for_cache['jac'] shape: {for_cache['jac'].shape}")
                print(f"  jac_for_original shape: {jac_for_original.shape}")
                print(f"  mixed_hess shape: {xccy_curve._mixed_hess_foreign_basis.shape}")
                print(f"  for_times: {for_times}")
                print(f"  xccy_times: {xccy_times}")

                # Get mixed Hessian and skip prepended points
                # Shape: [n_xccy_dfs, n_basis, n_for_dfs]
                # Note: JAX's jacfwd(jacrev(f, argnums=1), argnums=0) gives shape [output, arg1, arg0]
                mixed_hess_xccy = xccy_curve._mixed_hess_foreign_basis

                # Skip first row if XCCY has prepended t≈0
                if xccy_times[0] < 1e-6:
                    mixed_hess_xccy = mixed_hess_xccy[1:, :, :]

                # Skip first column of 3rd dimension if foreign curve has prepended t≈0
                if for_times[0] < 1e-6:
                    mixed_hess_xccy = mixed_hess_xccy[:, :, 1:]

                # Ensure third dimension matches jac_for_original's first dimension
                # The foreign curve might have extra points (e.g., spot_days=0 creates value date point)
                n_for_dfs_expected = jac_for_original.shape[0]
                n_for_dfs_actual = mixed_hess_xccy.shape[2]
                if n_for_dfs_actual > n_for_dfs_expected:
                    # Skip additional points from the beginning
                    skip_count = n_for_dfs_actual - n_for_dfs_expected
                    mixed_hess_xccy = mixed_hess_xccy[:, :, skip_count:]

                # Now mixed_hess_xccy shape is [n_xccy_dfs, n_basis, n_for_dfs]
                # where n_for_dfs matches jac_for_original's first dimension

                # Chain rule for cross-gamma: d²PV / d(for_rates) d(basis)
                # The mixed Hessian has shape [n_xccy_dfs, n_basis, n_for_dfs] from JAX
                # We need to chain with:
                # - PV gradient w.r.t. XCCY DFs: grad_xccy_dfs_original[i]
                # - Foreign OIS Jacobian (DFs to rates): jac_for_original[j,l]
                #
                # Result[k, l] = sum_i sum_j grad[i] * mixed_hess[i,k,j] * jac_for[j,l]
                # where i=xccy_dfs, k=basis, j=for_dfs, l=for_rates
                term1_cross = jnp.einsum('i,ikj,jl->kl',
                                         grad_xccy_dfs_original,  # [n_xccy_dfs]
                                         mixed_hess_xccy,          # [n_xccy_dfs, n_basis, n_for_dfs]
                                         jac_for_original)         # [n_for_dfs, n_for_rates]

                # For cross-gamma, we only need term1 (the mixed Hessian term)
                # Term2 is not applicable here because jac_xccy_for_ois is at payment level,
                # not pillar level, and cannot be chained with jac_for_original
                # Total cross-gamma matrix [n_basis, n_for_rates], transpose to [n_for_rates, n_basis]
                gamma_cross_matrix = term1_cross.T

                # Convert to GBP per bp2
                gamma_cross_matrix = gamma_cross_matrix * 1e-8 / spot_fx

                # Create CrossGamma object (expects [n_for_rates, n_basis])
                cross_gamma_for_basis = CrossGamma(
                    risk_matrix=gamma_cross_matrix,
                    tenors_curve1=to_tenor(foreign_model.swap_times),
                    tenors_curve2=basis_swap_tenors,
                    curve_type_1=derivative._foreign_floating_index,
                    curve_type_2=CurveTypes.USD_GBP_BASIS,
                    currency=derivative._domestic_currency
                )

            # Package gammas into Risk object with cross-gammas
            # Include XCCY basis gamma if available
            cross_gammas_list = [cross_gamma_for_basis] if cross_gamma_for_basis is not None else None

            if gamma_basis is not None:
                gamma = Risk([gamma_domestic, gamma_foreign, gamma_basis], cross_gammas=cross_gammas_list)
            else:
                gamma = Risk([gamma_domestic, gamma_foreign], cross_gammas=cross_gammas_list)

        # Cashflows extraction
        cashflows = None
        if RequestTypes.CASHFLOWS in reqs:
            all_cashflows = []

            # Extract domestic leg cashflows
            if hasattr(derivative, '_domestic_leg'):
                domestic_leg_type = "Domestic_Pay" if derivative._domestic_leg._leg_type == SwapTypes.PAY else "Domestic_Rec"
                domestic_cfs = self._extract_leg_cashflows(derivative._domestic_leg, domestic_leg_type)
                all_cashflows.extend(domestic_cfs)

            # Extract foreign leg cashflows
            if hasattr(derivative, '_foreign_leg'):
                foreign_leg_type = "Foreign_Rec" if derivative._domestic_leg._leg_type == SwapTypes.PAY else "Foreign_Pay"
                foreign_cfs = self._extract_leg_cashflows(derivative._foreign_leg, foreign_leg_type)
                all_cashflows.extend(foreign_cfs)

            cashflows = Cashflows(all_cashflows, risk_ccy)

        return AnalyticsResult(value=value, risk=delta, gamma=gamma, cashflows=cashflows)

    def _compute_xccy_old(self, derivative, reqs):
        """Old array-based implementation - kept for DELTA/GAMMA future work."""
        # Get curves from model
        domestic_model = getattr(self.model.curves, derivative._domestic_floating_index.name)
        foreign_model = getattr(self.model.curves, derivative._foreign_floating_index.name)

        # Get XCCY curve and spot FX
        foreign_code = derivative._foreign_currency.name
        domestic_code = derivative._domestic_currency.name
        xccy_curve_name = f"{foreign_code}_{domestic_code}_BASIS"

        try:
            xccy_curve = getattr(self.model.curves, xccy_curve_name)
            spot_fx = xccy_curve._spot_fx
        except AttributeError:
            raise LibError(f"XCCY curve {xccy_curve_name} not found in model. "
                         f"Available curves: {[attr for attr in dir(self.model.curves) if not attr.startswith('_')]}")

        # Detect leg types to route appropriately
        is_domestic_fixed = isinstance(derivative._domestic_leg, SwapFixedLeg)
        is_foreign_fixed = isinstance(derivative._foreign_leg, SwapFixedLeg)

        # Compute domestic leg analytics
        if is_domestic_fixed:
            # Fixed leg: use fixed leg analytics
            domestic_analytics = self._fixed_leg_analytics(
                domestic_model.swap_rates,
                domestic_model.swap_times,
                domestic_model.year_fracs,
                derivative._domestic_leg,
                domestic_model._value_dt,
                domestic_model._interp_type,
                reqs
            )
            # Add notional exchanges for fixed legs
            if RequestTypes.VALUE in reqs:
                notional_analytics = self._notional_exchange_value(
                    domestic_model.swap_rates,
                    domestic_model.swap_times,
                    domestic_model.year_fracs,
                    derivative._effective_dt,
                    derivative._maturity_dt,
                    derivative._domestic_notional,
                    domestic_model._value_dt,
                    domestic_model._interp_type,
                    derivative._domestic_currency,
                    derivative._domestic_floating_index,
                    derivative._domestic_leg_type
                )
                domestic_value = domestic_analytics["value"].amount + notional_analytics["value"].amount
                domestic_analytics["value"] = Valuation(amount=domestic_value, currency=derivative._domestic_currency)
        else:
            # Floating leg: use XCCY floating leg analytics (handles notional exchanges)
            if derivative._domestic_leg._notional_exchange:
                domestic_analytics = self._xccy_float_leg_analytics(
                    domestic_model.swap_rates,
                    domestic_model.swap_times,
                    domestic_model.year_fracs,
                    derivative._domestic_leg,
                    domestic_model._value_dt,
                    domestic_model._interp_type,  # discount curve
                    domestic_model._interp_type,  # index curve
                    None,  # first_fixing_rate
                    reqs,
                    derivative._effective_dt,
                    derivative._maturity_dt
                )
            else:
                domestic_analytics = self._float_leg_analytics(
                    domestic_model.swap_rates,
                    domestic_model.swap_times,
                    domestic_model.year_fracs,
                    derivative._domestic_leg,
                    domestic_model._value_dt,
                    domestic_model._interp_type,  # discount curve
                    domestic_model._interp_type,  # index curve
                    None,  # first_fixing_rate
                    reqs,
                )

        # Compute foreign leg analytics
        if is_foreign_fixed:
            # Fixed leg: use fixed leg analytics
            foreign_analytics = self._fixed_leg_analytics(
                foreign_model.swap_rates,
                foreign_model.swap_times,
                foreign_model.year_fracs,
                derivative._foreign_leg,
                foreign_model._value_dt,
                foreign_model._interp_type,
                reqs
            )
            # Add notional exchanges for fixed legs
            if RequestTypes.VALUE in reqs:
                notional_analytics = self._notional_exchange_value(
                    foreign_model.swap_rates,
                    foreign_model.swap_times,
                    foreign_model.year_fracs,
                    derivative._effective_dt,
                    derivative._maturity_dt,
                    derivative._foreign_notional,
                    foreign_model._value_dt,
                    foreign_model._interp_type,
                    derivative._foreign_currency,
                    derivative._foreign_floating_index,
                    derivative._foreign_leg._leg_type
                )
                foreign_value = foreign_analytics["value"].amount + notional_analytics["value"].amount
                foreign_analytics["value"] = Valuation(amount=foreign_value, currency=derivative._foreign_currency)
        else:
            # Floating leg: use XCCY curve for discounting, foreign curve for forward rates
            # The foreign leg coupons are projected using foreign OIS curve, but discounted using XCCY curve
            if derivative._foreign_leg._notional_exchange:
                foreign_analytics = self._xccy_float_leg_analytics(
                    foreign_model.swap_rates,
                    foreign_model.swap_times,
                    foreign_model.year_fracs,
                    derivative._foreign_leg,
                    foreign_model._value_dt,
                    xccy_curve,  # Pass XCCY curve object for discounting
                    foreign_model._interp_type,  # index curve type (for forward rates)
                    None,  # first_fixing_rate
                    reqs,
                    derivative._effective_dt,
                    derivative._maturity_dt
                )
            else:
                foreign_analytics = self._float_leg_analytics(
                    foreign_model.swap_rates,
                    foreign_model.swap_times,
                    foreign_model.year_fracs,
                    derivative._foreign_leg,
                    foreign_model._value_dt,
                    xccy_curve,  # Pass XCCY curve object for discounting
                    foreign_model._interp_type,  # index curve type (for forward rates)
                    None,  # first_fixing_rate
                    reqs,
                )

        # Combine results
        value = None
        if RequestTypes.VALUE in reqs:
            domestic_value = domestic_analytics["value"].amount
            foreign_value = foreign_analytics["value"].amount
            # Total PV = domestic PV + spot_FX * foreign PV (converted to domestic currency)
            total_value = domestic_value + foreign_value / spot_fx
            value = Valuation(amount=total_value, currency=derivative._domestic_currency)

        # DELTA and GAMMA not yet implemented for XCCY
        delta = None
        gamma = None

        # Cashflows extraction (placeholder for future implementation)
        cashflows = None
        if RequestTypes.CASHFLOWS in reqs:
            # TODO: Extract cashflow data from domestic and foreign legs
            cashflows = Cashflows([], derivative._domestic_currency)

        return AnalyticsResult(value=value, risk=delta, gamma=gamma, cashflows=cashflows)

    def valuation(self,
                  derivative):

        if derivative.derivative_type == InstrumentTypes.OIS_SWAP:

            ir_model = getattr(self.model.curves, derivative._floating_index.name)

            fixed_value = self.valuation_fixed_leg(
                    ir_model.swap_rates, 
                    ir_model.swap_times, 
                    ir_model.year_fracs,
                    derivative._fixed_leg,
                    ir_model._value_dt,
                    ir_model._interp_type
            )

            floating_value = self.valuation_float_leg(
                    swap_rates = ir_model.swap_rates, 
                    swap_times = ir_model.swap_times, 
                    year_fracs = ir_model.year_fracs,
                    floating_leg_details = derivative._float_leg,
                    value_dt = ir_model._value_dt,
                    discount_curve_type = ir_model._interp_type,
                    index_curve_type = ir_model._interp_type,
                    first_fixing_rate = None)
            
            return fixed_value + floating_value

        else:
            raise LibError(f"{self.derivative.derivative_type} not yet implemented")
        
    def delta(self,
                  derivative):

        if derivative.derivative_type == InstrumentTypes.OIS_SWAP: 

            ir_model = getattr(self.model.curves, derivative._floating_index.name)


            fixed_risk = self.delta_fixed_leg(
                    ir_model.swap_rates, 
                    ir_model.swap_times, 
                    ir_model.year_fracs,
                    derivative._fixed_leg,
                    ir_model._value_dt,
                    ir_model._interp_type
            )

            floating_risk = self.delta_float_leg(
                    ir_model.swap_rates, 
                    ir_model.swap_times, 
                    ir_model.year_fracs,
                    derivative._float_leg,
                    ir_model._value_dt,
                    ir_model._interp_type,
                    ir_model._interp_type,
                    None)
            
            return fixed_risk + floating_risk

        else:
            raise LibError(f"{self.derivative.derivative_type} not yet implemented")

    def gamma(self,
                derivative):

        if derivative.derivative_type == InstrumentTypes.OIS_SWAP: 

            ir_model = getattr(self.model.curves, derivative._floating_index.name)


            fixed_gamma = self.gamma_fixed_leg(
                    ir_model.swap_rates, 
                    ir_model.swap_times, 
                    ir_model.year_fracs,
                    derivative._fixed_leg,
                    ir_model._value_dt,
                    ir_model._interp_type
            )

            floating_gamma = self.gamma_float_leg(
                    ir_model.swap_rates, 
                    ir_model.swap_times, 
                    ir_model.year_fracs,
                    derivative._float_leg,
                    ir_model._value_dt,
                    ir_model._interp_type,
                    ir_model._interp_type,
                    None)
            
            return fixed_gamma + floating_gamma

        else:
            raise LibError(f"{self.derivative.derivative_type} not yet implemented")
    

    def build_curve_ad(self,
                    swap_rates: list[float],
                    swap_times: list[float],
                    year_fracs: list[list[float]]
                    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Bootstraps an OIS curve via par-swap rates using JAX-compatible operations.

        Matches the recursive logic from ois_curve.py by pre-expanding all intermediate
        cashflow points (not just swap maturities) and deduplicating using rounded keys.

        Args:
            swap_rates (list[float]): Par swap rates for each maturity
            swap_times (list[float]): Swap maturities in years
            year_fracs (list[list[float]]): Year fractions for each swap's cashflows

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]:
                - all_maturities: All unique intermediate times including t=0
                - all_dfs: Discount factors at all intermediate times (df=1.0 at t=0)

        Implementation:
            1. Pre-expands all intermediate cashflow points from all swaps
            2. Deduplicates using rounded maturity keys (2 decimal places)
            3. Builds dependency graph via prev_idx mapping
            4. Sequential bootstrap via lax.scan
            5. Prepends t=0, df=1.0 to match OISCurve behavior
            6. Returns dense grid for interpolation (not just swap maturities)

        Note:
            Each intermediate point inherits its parent swap's rate, matching
            the recursive version's behavior. Rounding to 2 decimals supports
            quarterly swaps (0.25) and is used only for dictionary key matching,
            not for actual computations.
        """

        # 1) Add t=0, df=1.0 point (matches OISCurve._build_curve_ad line 126-127)
        points = [{
            'maturity': 0.0,
            'maturity_key': 0.0,
            'acc': 0.0,  # No accrual for t=0 point
            'prev_mat': 0.0,
            'prev_key': None,  # No previous point
            'rate': swap_rates[0],  # Use first swap's rate (doesn't matter, df=1.0 is exact)
            'is_final': False,
            'swap_idx': -1  # Special index for t=0
        }]

        # 2) Pre-expand ALL intermediate points (not just swap maturities)
        for i, (rate, fracs) in enumerate(zip(swap_rates, year_fracs)):
            cumsum = 0.0
            for j, frac in enumerate(fracs):
                prev_cum = cumsum
                cumsum += frac
                points.append({
                    'maturity': cumsum,                      # EXACT value for computations
                    'maturity_key': round(cumsum, 2),       # Rounded key for matching (2 decimals to avoid collisions)
                    'acc': frac,
                    'prev_mat': prev_cum,                    # EXACT previous maturity
                    'prev_key': round(prev_cum, 2) if j > 0 else None,  # Rounded key (2 decimals)
                    'rate': rate,                            # Parent swap's rate
                    'is_final': (j == len(fracs) - 1),      # Is this the swap's final maturity?
                    'swap_idx': i
                })

        # 3) Sort by exact maturity
        sorted_points = sorted(points, key=lambda x: x['maturity'])

        # 4) Build maturity_key → first occurrence index mapping (for prev_idx lookup)
        # Like OISCurve's pv01_dict, we use rounded keys to find which previous point to use
        # But we keep ALL points in the arrays (no deduplication)
        maturity_lookup = {}  # rounded_key → FIRST index with this key
        for idx, p in enumerate(sorted_points):
            key = p['maturity_key']
            if key not in maturity_lookup:
                maturity_lookup[key] = idx  # Store first occurrence

        # 5) Build prev_idx for each point using rounded keys
        for p in sorted_points:
            if p['prev_key'] is None:
                p['prev_idx'] = -1
            else:
                p['prev_idx'] = maturity_lookup.get(p['prev_key'], -1)

        # 6) Convert to JAX arrays
        n_points = len(sorted_points)
        rates = jnp.array([p['rate'] for p in sorted_points])
        accs = jnp.array([p['acc'] for p in sorted_points])
        prev_idxs = jnp.array([p['prev_idx'] for p in sorted_points], dtype=jnp.int32)

        # 8) JAX-friendly scan through all points
        def step(pv01_arr, inputs):
            i, rate, acc, prev_idx = inputs
            prev_pv01 = jnp.where(prev_idx < 0, 0.0, pv01_arr[prev_idx])

            df_i = jnp.where(
                prev_idx < 0,
                1.0 / (1.0 + rate * acc),
                (1.0 - rate * prev_pv01) / (1.0 + rate * acc)
            )

            pv01_i = prev_pv01 + acc * df_i
            new_pv01 = pv01_arr.at[i].set(pv01_i)
            return new_pv01, df_i

        # 9) Run the scan
        init_pv01 = jnp.zeros(n_points)
        idxs = jnp.arange(n_points)
        _, all_dfs = lax.scan(step, init_pv01, (idxs, rates, accs, prev_idxs))

        # 10) Return ALL points (not just swap maturities)
        # This matches ois_curve.py which stores all intermediate DFs for interpolation
        # We keep ALL points (no deduplication) just like OISCurve
        all_maturities = jnp.array([p['maturity'] for p in sorted_points])
        return all_maturities, all_dfs

    def _cached_curve(self, key, swap_rates, swap_times, year_fracs, interp_type):
        """Bootstrap the curve once and cache DFS, Jacobian and Hessian."""
        cache = self._curve_cache.get(key)
        if cache is not None:
            return cache

        # Build curve to get both times and DFs (including all intermediate points)
        rates = jnp.array(swap_rates)
        times, dfs = self.build_curve_ad(rates, swap_times, year_fracs)

        # IMPORTANT: Prepend time≈0 with DF≈1.0 to enable forward rate calculations
        # from value date. Without this, interpolating start_times=0 fails.
        # Use time=1e-8 instead of exactly 0 to avoid numerical issues in FLAT_FWD_RATES
        # gradients (where rt = -log(DF) causes issues when DF=1.0 exactly).
        prepended_t0 = False
        if times[0] > 1e-7:
            times = jnp.concatenate([jnp.array([1e-8]), times])
            dfs = jnp.concatenate([jnp.array([1.0]), dfs])
            prepended_t0 = True

        # For AD, we need DFs as a function of rates only (times are constant)
        # Compute Jacobian for the original DFs (without time=0)
        def build_dfs_original(r):
            _, dfs_out = self.build_curve_ad(r, swap_times, year_fracs)
            return dfs_out

        jac_original = jacrev(build_dfs_original)(rates)
        hess_original = hessian(build_dfs_original)(rates)

        # If we prepended time=0, add a row of zeros to Jacobian and Hessian
        # because DF(t=0) = 1.0 is constant (zero gradient w.r.t. all rates)
        if prepended_t0:
            n_rates = len(rates)
            zero_row = jnp.zeros((1, n_rates))
            jac = jnp.concatenate([zero_row, jac_original], axis=0)

            # For Hessian: prepend zeros for the time=0 point
            zero_matrix = jnp.zeros((1, n_rates, n_rates))
            hess = jnp.concatenate([zero_matrix, hess_original], axis=0)
        else:
            jac = jac_original
            hess = hess_original

        cache = {
            "times": times,
            "dfs": dfs,
            "jac": jac,
            "hess": hess,
        }
        self._curve_cache[key] = cache
        return cache
        
    def _price_fixed_leg_jax(self,
                            dfs,
                            times,
                            interp_type,
                            payment_times,               # [M]
                            payments,                    # [M]
                            principal: float,            # scalar
                            notional: float,             # scalar
                            leg_sign: float,             # +1 or −1
                            value_time: float            # scalar
                            ):
        interp = InterpolatorAd(interp_type)
        df_val   = jnp.atleast_1d(interp.simple_interpolate(value_time, times, dfs, interp_type.value))
        df_pmts  = jnp.atleast_1d(interp.simple_interpolate(payment_times, times, dfs, interp_type.value))

        # build a mask of “after valuation date” over your M flows
        mask     = payment_times > value_time   # [M]
        # broadcast mask over any batch‐dimensions of df_pmts
        mask     = jnp.broadcast_to(mask, df_pmts.shape)

        # relative discount factors: shape [..., M]
        df_rel   = df_pmts / df_val[..., None]

        # PV of coupons:
        pv_coupons   = jnp.where(mask, payments * df_rel, 0.0)   # [..., M]
        # PV of final principal on last cash‐flow
        final_mask   = mask[..., -1]                            # [...]
        final_df_rel = df_rel[..., -1]                          # [...]
        pv_prin      = jnp.where(final_mask,
                                principal * final_df_rel,
                                0.0)                         # [...]

        # sum them up:
        leg_pv = jnp.sum(pv_coupons, axis=-1) + pv_prin          # [...]
        return leg_sign * leg_pv
    
    def value_fixed_leg(self,
                        swap_rates,
                        swap_times,
                        year_fracs,
                        fixed_leg_details,
                        value_dt: Date,
                        interpolator_dc_type):
        
        #swap_rates =  [x*1e-4  for x in swap_rates]

        curve_key = tuple(swap_times)
        cache = self._cached_curve(curve_key, swap_rates, swap_times, year_fracs, interpolator_dc_type)
        times = cache["times"]
        dfs = cache["dfs"]

        # — extract all the “static” pieces from your custom class ONCE —
        #    (these are plain Python numbers or JAX arrays)
        dc_type    = fixed_leg_details._dc_type
        # numeric offsets of each payment from the valuation date:
        payment_times = jnp.array([
            times_from_dates(dt, value_dt, dc_type)
            for dt in fixed_leg_details._payment_dts
        ])                               # shape [M]
        payments      = jnp.array(fixed_leg_details._payments)  # shape [M]
        principal     = fixed_leg_details._principal            # scalar
        notional      = fixed_leg_details._notional             # scalar
        leg_sign      = (
            +1.0 if fixed_leg_details._leg_type == SwapTypes.RECEIVE
            else -1.0
        )
        # numeric “value time”
        value_time = times_from_dates(value_dt, value_dt, dc_type)

        # — now call a tiny pure-JAX routine —
        pure_fn = partial(
            self._price_fixed_leg_jax,
            dfs=dfs,
            times=times,
            interp_type=interpolator_dc_type,
            payment_times=payment_times,
            payments=payments,
            principal=principal,
            notional=notional,
            leg_sign=leg_sign,
            value_time=value_time,
        )
        return pure_fn()

    def _fixed_leg_analytics(
        self,
        swap_rates,
        swap_times,
        year_fracs,
        fixed_leg_details,
        value_dt: Date,
        interpolator_dc_type,
        requests,
    ):
        """Common routine for PV/Delta/Gamma of the fixed leg."""

        curve_key = tuple(swap_times)
        cache = self._cached_curve(
            curve_key, swap_rates, swap_times, year_fracs, interpolator_dc_type
        )
        times = cache["times"]
        dfs = cache["dfs"]
        jac = cache["jac"]
        hess_curve = cache["hess"]

        dc_type = fixed_leg_details._dc_type
        payment_times = jnp.array(
            [times_from_dates(dt, value_dt, dc_type) for dt in fixed_leg_details._payment_dts]
        )
        payments = jnp.array(fixed_leg_details._payments)
        principal = fixed_leg_details._principal
        notional = fixed_leg_details._notional
        leg_sign = +1.0 if fixed_leg_details._leg_type == SwapTypes.RECEIVE else -1.0
        value_time = times_from_dates(value_dt, value_dt, dc_type)

        pv_fn = partial(
            self._price_fixed_leg_jax,
            times=times,
            interp_type=interpolator_dc_type,
            payment_times=payment_times,
            payments=payments,
            principal=principal,
            notional=notional,
            leg_sign=leg_sign,
            value_time=value_time,
        )

        out = {}
        if RequestTypes.VALUE in requests:
            val = pv_fn(dfs)
            # Convert to scalar - handles both scalar and (1,) array cases
            val_scalar = float(jnp.atleast_1d(val).item() if jnp.ndim(val) == 0 else val.squeeze())
            out["value"] = Valuation(amount=val_scalar, currency=fixed_leg_details._currency)

        need_grad = RequestTypes.DELTA in requests or RequestTypes.GAMMA in requests
        grad_dfs = None
        if need_grad:
            grad_dfs = grad(lambda d: jnp.squeeze(pv_fn(d)))(dfs)

        if RequestTypes.DELTA in requests:
            sensitivities = jnp.dot(grad_dfs, jac)
            sensies = [float(x) * 1e-4 for x in sensitivities]
            out["delta"] = Delta(
                risk_ladder=sensies,
                tenors=to_tenor(swap_times),
                currency=fixed_leg_details._currency,
                curve_type=fixed_leg_details._floating_index,
            )

        if RequestTypes.GAMMA in requests:
            hess_dfs = hessian(lambda d: jnp.squeeze(pv_fn(d)))(dfs)
            term1 = jac.T @ hess_dfs @ jac
            term2 = jnp.sum(grad_dfs[:, None, None] * hess_curve, axis=0)
            gammas = term1 + term2
            gammas = np.array(gammas, dtype=np.float64) * 1e-8
            out["gamma"] = Gamma(
                risk_ladder=gammas,
                tenors=to_tenor(swap_times),
                currency=fixed_leg_details._currency,
                curve_type=fixed_leg_details._floating_index,
            )

        return out

    def valuation_fixed_leg(
        self,
        swap_rates,
        swap_times,
        year_fracs,
        fixed_leg_details,
        value_dt: Date,
        interpolator_dc_type,
    ):
        res = self._fixed_leg_analytics(
            swap_rates,
            swap_times,
            year_fracs,
            fixed_leg_details,
            value_dt,
            interpolator_dc_type,
            {RequestTypes.VALUE},
        )
        return res["value"]

    def delta_fixed_leg(
        self,
        swap_rates,
        swap_times,
        year_fracs,
        fixed_leg_details,
        value_dt: Date,
        interpolator_dc_type,
    ):
        res = self._fixed_leg_analytics(
            swap_rates,
            swap_times,
            year_fracs,
            fixed_leg_details,
            value_dt,
            interpolator_dc_type,
            {RequestTypes.DELTA},
        )
        return res["delta"]

    def gamma_fixed_leg(
        self,
        swap_rates,
        swap_times,
        year_fracs,
        fixed_leg_details,
        value_dt: Date,
        interpolator_dc_type,
    ):
        res = self._fixed_leg_analytics(
            swap_rates,
            swap_times,
            year_fracs,
            fixed_leg_details,
            value_dt,
            interpolator_dc_type,
            {RequestTypes.GAMMA},
        )
        return res["gamma"]
    

    def _float_leg_jax(self,
                    dfs,
                    times,
                    disc_interp_type,
                    idx_interp_type,
                    payment_times,                # [M]
                    start_times,                  # [M]
                    end_times,                    # [M]
                    pay_alphas,                   # [M]
                    spreads,                      # [M]
                    notionals,                    # [M]
                    principal: float,             # scalar
                    leg_sign: float,              # +1 or –1
                    value_time: float,            # scalar
                    first_fixing_rate: float,
                    override_first,               # scalar
                    idx_times=None,               # Optional separate index curve times
                    idx_dfs=None,                 # Optional separate index curve dfs
                    notional_exchange=False,      # Optional: enable notional exchanges (for XCCY)
                    notional_exchange_amount=0.0, # Optional: notional amount to exchange
                    effective_time=0.0,           # Optional: time to effective date
                    maturity_time=0.0             # Optional: time to maturity date
                    ):
        disc_interp = InterpolatorAd(disc_interp_type)
        idx_interp  = InterpolatorAd(idx_interp_type)

        # Use separate index curve times/dfs if provided (for XCCY swaps)
        idx_times_actual = idx_times if idx_times is not None else times
        idx_dfs_actual = idx_dfs if idx_dfs is not None else dfs

        df_val   = jnp.atleast_1d(disc_interp.simple_interpolate(value_time, times, dfs, disc_interp_type.value))
        df_start = jnp.atleast_1d(idx_interp.simple_interpolate(start_times, idx_times_actual, idx_dfs_actual, idx_interp_type.value))
        df_end   = jnp.atleast_1d(idx_interp.simple_interpolate(end_times, idx_times_actual, idx_dfs_actual, idx_interp_type.value))

        # d) Vectorised forward rates
        # Avoid 0/0 when pay_alphas is zero (notional exchanges with no accrual period)
        fwd = jnp.where(pay_alphas > 0, (df_start / df_end - 1.0) / pay_alphas, 0.0)  # [..., M]

        # only override if the I actually passed a first_fixing_rate
        first_mask     = jnp.arange(fwd.shape[-1]) == 0             # [M]
        # make it match the batch dims
        first_mask_b   = jnp.broadcast_to(first_mask, fwd.shape)    # [..., M]

        # broadcast the static Python bool as well
        override_mask  = first_mask_b & override_first              # [..., M]

        # apply override only where override_mask is True
        fwd = jnp.where(override_mask, first_fixing_rate, fwd)      # [..., M]

        # e) coupon amounts
        cf_amounts = (fwd + spreads) * pay_alphas * notionals                    # [..., M]

        df_pmts    = jnp.atleast_1d(disc_interp.simple_interpolate(payment_times, times, dfs, disc_interp_type.value))
        df_rel     = df_pmts / df_val[..., None]                                 # [..., M]

        # g) mask out past payments
        valid      = payment_times >= value_time                                 # [M]
        valid      = jnp.broadcast_to(valid, cf_amounts.shape)                  # [..., M]

        # h) PV of coupons + principal
        pv_coupons = jnp.where(valid, cf_amounts * df_rel, 0.0)                  # [..., M]
        pv_prin    = jnp.where(valid[..., -1],
                            principal * df_rel[..., -1],
                            0.0)                                            # [...]

        # i) Notional exchanges (for XCCY swaps)
        # At effective_dt: -notional (outflow), at maturity_dt: +notional (inflow)
        pv_notional_exchange = 0.0
        if notional_exchange:
            # Start exchange: -notional at effective_dt (if in future or today)
            df_effective = jnp.atleast_1d(disc_interp.simple_interpolate(effective_time, times, dfs, disc_interp_type.value))
            df_effective_rel = df_effective / df_val
            pv_start_exchange = jnp.where(effective_time >= value_time,
                                         -notional_exchange_amount * df_effective_rel,
                                         0.0)

            # End exchange: +notional at maturity_dt (if in future)
            df_maturity = jnp.atleast_1d(disc_interp.simple_interpolate(maturity_time, times, dfs, disc_interp_type.value))
            df_maturity_rel = df_maturity / df_val
            pv_end_exchange = jnp.where(maturity_time >= value_time,
                                       notional_exchange_amount * df_maturity_rel,
                                       0.0)

            pv_notional_exchange = jnp.squeeze(pv_start_exchange + pv_end_exchange)

        # j) aggregate and apply sign
        pv_coupons_sum = jnp.sum(pv_coupons, axis=-1)
        leg_pv     = pv_coupons_sum + pv_prin + pv_notional_exchange  # [...]

        return leg_sign * leg_pv
            
    def value_float_leg(self,
                    swap_rates,
                    swap_times,
                    year_fracs,
                    floating_leg_details,
                    value_dt,
                    discount_curve_type,
                    index_curve_type = None,
                    first_fixing_rate = None):
        """
        Compute the floating‐leg PV, building both discount and index
        InterpolatorAd() objects from their interp‐type strings.
        """
        curve_key = tuple(swap_times)
        cache = self._cached_curve(curve_key, swap_rates, swap_times, year_fracs, discount_curve_type)
        times = cache["times"]
        dfs = cache["dfs"]

        # 2) Build (or default) the index curve
        if index_curve_type is None:
            idx_interp = disc_interp
        else:
            idx_interp = InterpolatorAd(index_curve_type)
            idx_interp.fit(times=times, dfs=dfs)

        # 3) Extract all “static” inputs from your custom class & value_dt
        dc_type      = floating_leg_details._dc_type
        # payment, start, end offsets from value_dt → [M]
        payment_times = jnp.array([
            times_from_dates(dt, value_dt, dc_type)
            for dt in floating_leg_details._payment_dts
        ])
        start_times   = jnp.array([
            times_from_dates(dt0, value_dt, dc_type)
            for dt0 in floating_leg_details._start_accrued_dts
        ])
        end_times     = jnp.array([
            times_from_dates(dt1, value_dt, dc_type)
            for dt1 in floating_leg_details._end_accrued_dts
        ])

        pay_alphas    = jnp.array(floating_leg_details._year_fracs)        # [M]
        spreads       = jnp.full_like(pay_alphas, floating_leg_details._spread)
        notionals     = jnp.array(
            floating_leg_details._notional_array
            or [floating_leg_details._notional] * len(pay_alphas)
        )                                                                   # [M]

        principal     = floating_leg_details._principal                     # scalar
        leg_sign      = (+1.0 
                        if floating_leg_details._leg_type == SwapTypes.RECEIVE
                        else -1.0)
        value_time    = times_from_dates(value_dt, value_dt, dc_type)      # scalar
        override_first = first_fixing_rate is not None
        fix0          = first_fixing_rate if override_first else 0.0
        #fix0          = first_fixing_rate or 0.0                           # scalar

        pure_fn = partial(
            self._float_leg_jax,
            dfs=dfs,
            times=times,
            disc_interp_type=discount_curve_type,
            idx_interp_type=index_curve_type or discount_curve_type,
            payment_times=payment_times,
            start_times=start_times,
            end_times=end_times,
            pay_alphas=pay_alphas,
            spreads=spreads,
            notionals=notionals,
            principal=principal,
            leg_sign=leg_sign,
            value_time=value_time,
            first_fixing_rate=fix0,
            override_first=override_first,
        )

        return pure_fn()
    
    def _float_leg_analytics(
        self,
        swap_rates,
        swap_times,
        year_fracs,
        floating_leg_details,
        value_dt,
        discount_curve_type,
        index_curve_type=None,
        first_fixing_rate=None,
        requests=None,
    ):
        """Common routine for PV/Delta/Gamma of the floating leg."""

        if requests is None:
            requests = {RequestTypes.VALUE}

        # Check if discount_curve_type is an actual curve object (XccyCurve)
        from cavour.trades.rates.xccy_curve import XccyCurve
        idx_times = None
        idx_dfs = None

        if isinstance(discount_curve_type, XccyCurve):
            # Use pre-computed times and dfs from XCCY curve for discounting
            times = jnp.array(discount_curve_type._times)
            dfs = jnp.array(discount_curve_type._dfs)
            jac = None  # Not available for pre-computed curves
            hess_curve = None
            actual_interp_type = discount_curve_type._interp_type

            # For index curve (forward rates), use the provided swap_rates/swap_times
            # These come from the foreign OIS curve for XCCY swaps
            idx_curve_key = tuple(swap_times)
            idx_cache = self._cached_curve(
                idx_curve_key, swap_rates, swap_times, year_fracs, index_curve_type or actual_interp_type
            )
            idx_times = idx_cache["times"]
            idx_dfs = idx_cache["dfs"]
        else:
            # Normal case: bootstrap curve from rates
            curve_key = tuple(swap_times)
            cache = self._cached_curve(
                curve_key, swap_rates, swap_times, year_fracs, discount_curve_type
            )
            times = cache["times"]
            dfs = cache["dfs"]
            jac = cache["jac"]
            hess_curve = cache["hess"]
            actual_interp_type = discount_curve_type

        dc_type = floating_leg_details._dc_type
        payment_times = jnp.array(
            [times_from_dates(dt, value_dt, dc_type) for dt in floating_leg_details._payment_dts]
        )
        start_times = jnp.array(
            [times_from_dates(dt0, value_dt, dc_type) for dt0 in floating_leg_details._start_accrued_dts]
        )
        end_times = jnp.array(
            [times_from_dates(dt1, value_dt, dc_type) for dt1 in floating_leg_details._end_accrued_dts]
        )
        pay_alphas = jnp.array(floating_leg_details._year_fracs)
        spreads = jnp.full_like(pay_alphas, floating_leg_details._spread)
        notionals = jnp.array(
            floating_leg_details._notional_array or [floating_leg_details._notional] * len(pay_alphas)
        )
        principal = floating_leg_details._principal
        leg_sign = +1.0 if floating_leg_details._leg_type == SwapTypes.RECEIVE else -1.0
        value_time = times_from_dates(value_dt, value_dt, dc_type)
        override_first = first_fixing_rate is not None
        fix0 = first_fixing_rate if override_first else 0.0

        pv_fn = partial(
            self._float_leg_jax,
            times=times,
            disc_interp_type=actual_interp_type,
            idx_interp_type=index_curve_type or actual_interp_type,
            payment_times=payment_times,
            start_times=start_times,
            end_times=end_times,
            pay_alphas=pay_alphas,
            spreads=spreads,
            notionals=notionals,
            principal=principal,
            leg_sign=leg_sign,
            value_time=value_time,
            first_fixing_rate=fix0,
            override_first=override_first,
            idx_times=idx_times,  # For XCCY: separate index curve
            idx_dfs=idx_dfs,
        )

        out = {}
        if RequestTypes.VALUE in requests:
            val = pv_fn(dfs)
            # Convert to scalar - handles both scalar and (1,) array cases
            val_scalar = float(jnp.atleast_1d(val).item() if jnp.ndim(val) == 0 else val.squeeze())
            out["value"] = Valuation(amount=val_scalar, currency=floating_leg_details._currency)

        need_grad = RequestTypes.DELTA in requests or RequestTypes.GAMMA in requests
        grad_dfs = None
        if need_grad:
            grad_dfs = grad(lambda d: jnp.squeeze(pv_fn(d)))(dfs)

        if RequestTypes.DELTA in requests:
            sensitivities = jnp.dot(grad_dfs, jac)
            sensies = [float(x) * 1e-4 for x in sensitivities]
            out["delta"] = Delta(
                risk_ladder=sensies,
                tenors=to_tenor(swap_times),
                currency=floating_leg_details._currency,
                curve_type=floating_leg_details._floating_index,
            )

        if RequestTypes.GAMMA in requests:
            hess_dfs = hessian(lambda d: jnp.squeeze(pv_fn(d)))(dfs)
            term1 = jac.T @ hess_dfs @ jac
            term2 = jnp.sum(grad_dfs[:, None, None] * hess_curve, axis=0)
            gammas = term1 + term2
            gammas = np.array(gammas, dtype=np.float64) * 1e-8
            out["gamma"] = Gamma(
                risk_ladder=gammas,
                tenors=to_tenor(swap_times),
                currency=floating_leg_details._currency,
                curve_type=floating_leg_details._floating_index,
            )

        return out

    def _xccy_float_leg_analytics(
        self,
        swap_rates,
        swap_times,
        year_fracs,
        floating_leg_details,
        value_dt,
        discount_curve_type,
        index_curve_type,
        first_fixing_rate,
        requests,
        effective_dt,
        maturity_dt
    ):
        """
        Compute analytics for XCCY floating leg with notional exchanges.

        This extends _float_leg_analytics to handle notional exchanges which
        are critical for XCCY swaps but not standard OIS swaps.

        Args:
            swap_rates: Par rates for curve building
            swap_times: Swap maturities
            year_fracs: Year fractions for each swap
            floating_leg_details: SwapFloatLeg instance
            value_dt: Valuation date
            discount_curve_type: Interpolation type for discounting
            index_curve_type: Interpolation type for forward rates
            first_fixing_rate: Optional first fixing rate
            requests: Set of RequestTypes (VALUE, DELTA, GAMMA)
            effective_dt: Swap effective date
            maturity_dt: Swap maturity date

        Returns:
            dict with value, delta, gamma (VALUE only for now)
        """
        # Get coupon cashflows analytics using standard method
        coupon_analytics = self._float_leg_analytics(
            swap_rates,
            swap_times,
            year_fracs,
            floating_leg_details,
            value_dt,
            discount_curve_type,
            index_curve_type,
            first_fixing_rate,
            requests
        )

        # Get notional exchange analytics
        # For XCCY curves, pass empty arrays since we use pre-computed times/dfs
        from cavour.trades.rates.xccy_curve import XccyCurve
        if isinstance(discount_curve_type, XccyCurve):
            # Use empty arrays - _notional_exchange_value will detect XCCY curve
            notional_swap_rates = jnp.array([])
            notional_swap_times = jnp.array([])
            notional_year_fracs = jnp.array([])
        else:
            # Normal case: use provided data
            notional_swap_rates = swap_rates
            notional_swap_times = swap_times
            notional_year_fracs = year_fracs

        notional_analytics = self._notional_exchange_value(
            notional_swap_rates,
            notional_swap_times,
            notional_year_fracs,
            effective_dt,
            maturity_dt,
            floating_leg_details._notional,
            value_dt,
            discount_curve_type,
            floating_leg_details._currency,
            floating_leg_details._floating_index,
            floating_leg_details._leg_type
        )

        # Combine results
        out = {}
        if RequestTypes.VALUE in requests:
            coupon_value = coupon_analytics.get("value").amount
            notional_value = notional_analytics.get("value").amount
            total_value = coupon_value + notional_value
            out["value"] = Valuation(amount=total_value, currency=floating_leg_details._currency)

        # TODO: DELTA and GAMMA will be added in Phase 2

        return out

    def _notional_exchange_value(
        self,
        swap_rates,
        swap_times,
        year_fracs,
        effective_dt,
        maturity_dt,
        notional,
        value_dt,
        interp_type,
        currency,
        curve_type,
        leg_type
    ):
        """
        Compute VALUE for notional exchanges at start and maturity.

        For XCCY swaps, notional is exchanged at:
        - Start (effective_dt): -notional (outflow)
        - Maturity: +notional (inflow)

        Args:
            swap_rates: Par rates for curve bootstrapping
            swap_times: Swap maturities
            year_fracs: Year fractions for curve building
            effective_dt: Swap effective date
            maturity_dt: Swap maturity date
            notional: Notional amount
            value_dt: Valuation date
            interp_type: Interpolation type
            currency: Currency for valuation
            curve_type: Curve type identifier
            leg_type: SwapTypes.RECEIVE or SwapTypes.PAY

        Returns:
            dict with 'value': Valuation object
        """
        # Check if interp_type is an actual curve object (XccyCurve)
        from cavour.trades.rates.xccy_curve import XccyCurve
        if isinstance(interp_type, XccyCurve):
            # Use pre-computed times and dfs from the curve
            times = jnp.array(interp_type._times)
            dfs = jnp.array(interp_type._dfs)
            actual_interp_type = interp_type._interp_type
        else:
            # Normal case: bootstrap curve from rates
            curve_key = tuple(swap_times)
            cache = self._cached_curve(curve_key, swap_rates, swap_times, year_fracs, interp_type)
            times = cache["times"]
            dfs = cache["dfs"]
            actual_interp_type = interp_type

        # Build interpolator
        from cavour.market.curves.interpolator_ad import InterpolatorAd
        interp = InterpolatorAd(actual_interp_type)

        dc_type = DayCountTypes.ACT_365F  # Standard for time calculations
        value_time = times_from_dates(value_dt, value_dt, dc_type)
        df_value = float(interp.simple_interpolate(value_time, times, dfs, actual_interp_type.value))
        total_value = 0.0

        # Start exchange: -notional at effective_dt (outflow)
        if effective_dt >= value_dt:
            effective_time = times_from_dates(effective_dt, value_dt, dc_type)
            df_start_abs = float(interp.simple_interpolate(effective_time, times, dfs, actual_interp_type.value))
            df_start = df_start_abs / df_value  # Normalize by DF at valuation date
            start_exchange_pv = -notional * df_start
            total_value += start_exchange_pv

        # End exchange: +notional at maturity_dt (inflow)
        if maturity_dt >= value_dt:
            maturity_time = times_from_dates(maturity_dt, value_dt, dc_type)
            df_end_abs = float(interp.simple_interpolate(maturity_time, times, dfs, actual_interp_type.value))
            df_end = df_end_abs / df_value  # Normalize by DF at valuation date
            end_exchange_pv = notional * df_end
            total_value += end_exchange_pv

        # Apply leg type sign
        if leg_type == SwapTypes.PAY:
            total_value = -total_value

        return {
            "value": Valuation(amount=total_value, currency=currency)
        }

    def valuation_float_leg(
        self,
        swap_rates,
        swap_times,
        year_fracs,
        floating_leg_details,
        value_dt,
        discount_curve_type,
        index_curve_type=None,
        first_fixing_rate=None,
    ):
        res = self._float_leg_analytics(
            swap_rates,
            swap_times,
            year_fracs,
            floating_leg_details,
            value_dt,
            discount_curve_type,
            index_curve_type,
            first_fixing_rate,
            {RequestTypes.VALUE},
        )
        return res["value"]

    def delta_float_leg(
        self,
        swap_rates,
        swap_times,
        year_fracs,
        floating_leg_details,
        value_dt,
        discount_curve_type,
        index_curve_type=None,
        first_fixing_rate=None,
    ):
        res = self._float_leg_analytics(
            swap_rates,
            swap_times,
            year_fracs,
            floating_leg_details,
            value_dt,
            discount_curve_type,
            index_curve_type,
            first_fixing_rate,
            {RequestTypes.DELTA},
        )
        return res["delta"]

    def gamma_float_leg(
        self,
        swap_rates,
        swap_times,
        year_fracs,
        floating_leg_details,
        value_dt,
        discount_curve_type,
        index_curve_type=None,
        first_fixing_rate=None,
    ):
        res = self._float_leg_analytics(
            swap_rates,
            swap_times,
            year_fracs,
            floating_leg_details,
            value_dt,
            discount_curve_type,
            index_curve_type,
            first_fixing_rate,
            {RequestTypes.GAMMA},
        )
        return res["gamma"]