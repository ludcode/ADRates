"Valuation Engine"

import numpy as np
import jax.numpy as jnp
from jax import lax, grad, hessian, jacrev, linearize, jit
from functools import partial
from typing import Sequence, Any, Dict

from cavour.market.curves.interpolator import *
from cavour.utils.helpers import to_tenor, times_from_dates
from cavour.utils.date import Date
from cavour.utils.day_count import DayCountTypes
from cavour.utils.error import LibError
from cavour.market.curves.interpolator_ad import InterpolatorAd
from cavour.requests.results import Valuation, Gamma, Delta, AnalyticsResult, Risk, CrossGamma, FXDelta, Cashflows, CashflowItem
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

        # Route Cash Deposits to deposit handler
        if derivative.derivative_type == InstrumentTypes.CASH_DEPOSIT:
            return self._compute_deposit(derivative, reqs, collateral_type)

        # Route FRAs to FRA handler
        if derivative.derivative_type == InstrumentTypes.FRA:
            return self._compute_fra(derivative, reqs, collateral_type)

        # Route IR Futures to futures handler
        if derivative.derivative_type == InstrumentTypes.STIR_FUTURE:
            return self._compute_stir_future(derivative, reqs, collateral_type)

        raise LibError(f"{derivative.derivative_type} not yet implemented")


    def compute_batch(self, derivatives, request_list, collateral_type=None):
        """Compute analytics for a batch of derivatives using JAX vectorization (vmap).

        This method provides 2-5x speedup for batches of 10+ swaps by:
        1. Extracting curve data once (shared across batch)
        2. Vectorizing the pricing computation via JAX vmap
        3. Avoiding repeated Python overhead (dictionary lookups, attribute access)

        Performance characteristics:
            - Homogeneous batches (same maturity): 3-5x speedup
            - Semi-heterogeneous (different start dates): 2-4x speedup
            - Heterogeneous (different maturities): 2-3x speedup
            - Very heterogeneous: 1.5-2x speedup

        Constraints:
            - All derivatives must be the same type (XCCY swaps or OIS swaps)
            - All derivatives must use the same curves (same currency pair for XCCY, same index for OIS)
            - All derivatives must have the same structure (frequency, daycount)

        Args:
            derivatives: List of derivative instruments (must be same type and structure)
            request_list: List of RequestTypes (VALUE, DELTA, GAMMA)
            collateral_type (CollateralType, optional): Not currently supported for batch

        Returns:
            List of AnalyticsResult objects (one per derivative, same order as input)

        Raises:
            LibError: If derivatives have incompatible types or structures
            NotImplementedError: If requested for unsupported derivatives or with collateral

        Example:
            >>> swaps = [create_xccy_swap(maturity=f'{i}Y') for i in range(5, 25, 5)]
            >>> results = engine.compute_batch(swaps, [RequestTypes.VALUE, RequestTypes.DELTA])
            >>> pvs = [r.value.amount for r in results]

        Note:
            For small batches (<10 swaps), sequential compute() may be faster due to
            vmap compilation overhead. Use compute_batch() for 10+ swaps.
        """
        from jax import vmap
        import jax.numpy as jnp
        from cavour.utils.helpers import times_from_dates
        from cavour.requests.results import AnalyticsResult, Valuation

        reqs = set(request_list)

        # Validate inputs
        if not derivatives:
            raise LibError("compute_batch requires at least one derivative")

        if collateral_type is not None:
            raise NotImplementedError("compute_batch does not yet support collateral_type")

        # Validate all derivatives are the same supported type
        first_type = derivatives[0].derivative_type
        if first_type not in [InstrumentTypes.XCCY_SWAP, InstrumentTypes.OIS_SWAP]:
            raise NotImplementedError(
                f"compute_batch currently supports XCCY swaps and OIS swaps, got {first_type}"
            )

        # Route to appropriate implementation
        if first_type == InstrumentTypes.XCCY_SWAP:
            return self._compute_batch_xccy(derivatives, reqs)
        elif first_type == InstrumentTypes.OIS_SWAP:
            return self._compute_batch_ois(derivatives, reqs)
        else:
            raise LibError(f"Unsupported derivative type: {first_type}")

    def _compute_batch_xccy(self, derivatives, reqs):
        """Internal method for batched XCCY swap computation."""
        from jax import vmap
        import jax.numpy as jnp
        from cavour.utils.helpers import times_from_dates
        from cavour.requests.results import AnalyticsResult, Valuation

        # Validate all derivatives are XCCY swaps
        for i, deriv in enumerate(derivatives):
            if deriv.derivative_type != InstrumentTypes.XCCY_SWAP:
                raise LibError(f"All derivatives must be XCCY swaps. derivatives[{i}] is {deriv.derivative_type}")

        # Validate all swaps use same curves (same currency pair)
        first = derivatives[0]
        dom_idx_name = first._domestic_floating_index.name
        for_idx_name = first._foreign_floating_index.name
        dom_ccy_name = first._domestic_currency.name
        for_ccy_name = first._foreign_currency.name

        for i, deriv in enumerate(derivatives[1:], 1):
            if (deriv._domestic_floating_index.name != dom_idx_name or
                deriv._foreign_floating_index.name != for_idx_name or
                deriv._domestic_currency.name != dom_ccy_name or
                deriv._foreign_currency.name != for_ccy_name):
                raise LibError(f"All derivatives must use same curves. derivatives[{i}] has incompatible curves")

        # Extract curves once (shared across batch)
        domestic_model = getattr(self.model.curves, dom_idx_name)
        foreign_model = getattr(self.model.curves, for_idx_name)
        xccy_curve_name = f"{for_ccy_name}_{dom_ccy_name}_BASIS"

        try:
            xccy_curve = getattr(self.model.curves, xccy_curve_name)
            spot_fx = xccy_curve._spot_fx
        except AttributeError:
            raise LibError(f"XCCY curve {xccy_curve_name} not found in model")

        # Prepare curve arrays (shared across batch)
        dom_times = jnp.array(domestic_model._times)
        dom_dfs = jnp.array(domestic_model._dfs)
        if dom_times[0] > 1e-7:
            dom_times = jnp.concatenate([jnp.array([1e-8]), dom_times])
            dom_dfs = jnp.concatenate([jnp.array([1.0]), dom_dfs])

        for_times = jnp.array(foreign_model._times)
        for_dfs = jnp.array(foreign_model._dfs)
        if for_times[0] > 1e-7:
            for_times = jnp.concatenate([jnp.array([1e-8]), for_times])
            for_dfs = jnp.concatenate([jnp.array([1.0]), for_dfs])

        xccy_times = jnp.array(xccy_curve._times)
        xccy_dfs = jnp.array(xccy_curve._dfs)

        # Extract leg parameters for each swap and determine max cashflows
        batch_size = len(derivatives)
        max_dom_cashflows = max(len(d._domestic_leg._payment_dts) for d in derivatives)
        max_for_cashflows = max(len(d._foreign_leg._payment_dts) for d in derivatives)

        # Preallocate padded arrays
        dom_payment_times_batch = jnp.zeros((batch_size, max_dom_cashflows))
        dom_start_times_batch = jnp.zeros((batch_size, max_dom_cashflows))
        dom_end_times_batch = jnp.zeros((batch_size, max_dom_cashflows))
        dom_alphas_batch = jnp.zeros((batch_size, max_dom_cashflows))
        dom_spreads_batch = jnp.zeros((batch_size, max_dom_cashflows))
        dom_notionals_batch = jnp.zeros((batch_size, max_dom_cashflows))

        for_payment_times_batch = jnp.zeros((batch_size, max_for_cashflows))
        for_start_times_batch = jnp.zeros((batch_size, max_for_cashflows))
        for_end_times_batch = jnp.zeros((batch_size, max_for_cashflows))
        for_alphas_batch = jnp.zeros((batch_size, max_for_cashflows))
        for_spreads_batch = jnp.zeros((batch_size, max_for_cashflows))
        for_notionals_batch = jnp.zeros((batch_size, max_for_cashflows))

        # Scalar arrays
        dom_principal_batch = jnp.zeros(batch_size)
        dom_leg_sign_batch = jnp.zeros(batch_size)
        dom_notional_exchange_batch = jnp.zeros(batch_size, dtype=bool)
        dom_effective_time_batch = jnp.zeros(batch_size)
        dom_maturity_time_batch = jnp.zeros(batch_size)

        for_principal_batch = jnp.zeros(batch_size)
        for_leg_sign_batch = jnp.zeros(batch_size)
        for_notional_exchange_batch = jnp.zeros(batch_size, dtype=bool)
        for_effective_time_batch = jnp.zeros(batch_size)
        for_maturity_time_batch = jnp.zeros(batch_size)

        # Extract and pad parameters for each swap
        dc_type = first._domestic_leg._dc_type
        value_time = times_from_dates(self.model.value_dt, self.model.value_dt, dc_type)

        for i, deriv in enumerate(derivatives):
            # Domestic leg
            n_dom = len(deriv._domestic_leg._payment_dts)
            dom_payment_times_batch = dom_payment_times_batch.at[i, :n_dom].set(
                jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                          for dt in deriv._domestic_leg._payment_dts]))
            dom_start_times_batch = dom_start_times_batch.at[i, :n_dom].set(
                jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                          for dt in deriv._domestic_leg._start_accrued_dts]))
            dom_end_times_batch = dom_end_times_batch.at[i, :n_dom].set(
                jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                          for dt in deriv._domestic_leg._end_accrued_dts]))
            dom_alphas_batch = dom_alphas_batch.at[i, :n_dom].set(
                jnp.array(deriv._domestic_leg._year_fracs))
            dom_spreads_batch = dom_spreads_batch.at[i, :n_dom].set(
                jnp.full(n_dom, deriv._domestic_leg._spread))
            dom_notionals_batch = dom_notionals_batch.at[i, :n_dom].set(
                jnp.array(deriv._domestic_leg._notional_array or
                         [deriv._domestic_leg._notional] * n_dom))

            dom_principal_batch = dom_principal_batch.at[i].set(deriv._domestic_leg._principal)
            dom_leg_sign_batch = dom_leg_sign_batch.at[i].set(
                +1.0 if deriv._domestic_leg._leg_type == SwapTypes.RECEIVE else -1.0)
            dom_notional_exchange_batch = dom_notional_exchange_batch.at[i].set(
                deriv._domestic_leg._notional_exchange)
            dom_effective_time_batch = dom_effective_time_batch.at[i].set(
                times_from_dates(deriv._effective_dt, self.model.value_dt, dc_type))
            dom_maturity_time_batch = dom_maturity_time_batch.at[i].set(
                times_from_dates(deriv._maturity_dt, self.model.value_dt, dc_type))

            # Foreign leg
            for_dc_type = deriv._foreign_leg._dc_type
            xccy_dc_type = xccy_curve._dc_type
            n_for = len(deriv._foreign_leg._payment_dts)

            for_payment_times_batch = for_payment_times_batch.at[i, :n_for].set(
                jnp.array([times_from_dates(dt, self.model.value_dt, xccy_dc_type)
                          for dt in deriv._foreign_leg._payment_dts]))
            for_start_times_batch = for_start_times_batch.at[i, :n_for].set(
                jnp.array([times_from_dates(dt, self.model.value_dt, for_dc_type)
                          for dt in deriv._foreign_leg._start_accrued_dts]))
            for_end_times_batch = for_end_times_batch.at[i, :n_for].set(
                jnp.array([times_from_dates(dt, self.model.value_dt, for_dc_type)
                          for dt in deriv._foreign_leg._end_accrued_dts]))
            for_alphas_batch = for_alphas_batch.at[i, :n_for].set(
                jnp.array(deriv._foreign_leg._year_fracs))
            for_spreads_batch = for_spreads_batch.at[i, :n_for].set(
                jnp.full(n_for, deriv._foreign_leg._spread))
            for_notionals_batch = for_notionals_batch.at[i, :n_for].set(
                jnp.array(deriv._foreign_leg._notional_array or
                         [deriv._foreign_leg._notional] * n_for))

            for_principal_batch = for_principal_batch.at[i].set(deriv._foreign_leg._principal)
            for_leg_sign_batch = for_leg_sign_batch.at[i].set(
                +1.0 if deriv._foreign_leg._leg_type == SwapTypes.RECEIVE else -1.0)
            for_notional_exchange_batch = for_notional_exchange_batch.at[i].set(
                deriv._foreign_leg._notional_exchange)
            for_effective_time_batch = for_effective_time_batch.at[i].set(
                times_from_dates(deriv._effective_dt, self.model.value_dt, xccy_dc_type))
            for_maturity_time_batch = for_maturity_time_batch.at[i].set(
                times_from_dates(deriv._maturity_dt, self.model.value_dt, xccy_dc_type))

        # Create vectorized PV function using vmap
        # in_axes: None = shared across batch, 0 = batched (one per swap)
        pv_batch_fn = vmap(
            lambda dom_pmt, dom_start, dom_end, dom_alpha, dom_spr, dom_not,
                   dom_prin, dom_sign, dom_notex, dom_eff, dom_mat,
                   for_pmt, for_start, for_end, for_alpha, for_spr, for_not,
                   for_prin, for_sign, for_notex, for_eff, for_mat:
                self._xccy_pv_pure(
                    dom_dfs, dom_times, domestic_model._interp_type,
                    for_dfs, for_times, foreign_model._interp_type,
                    xccy_dfs, xccy_times, xccy_curve._interp_type,
                    dom_pmt, dom_start, dom_end, dom_alpha, dom_spr, dom_not,
                    dom_prin, dom_sign, dom_notex, dom_eff, dom_mat,
                    for_pmt, for_start, for_end, for_alpha, for_spr, for_not,
                    for_prin, for_sign, for_notex, for_eff, for_mat,
                    value_time, spot_fx
                ),
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Domestic leg batched
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)   # Foreign leg batched
        )

        # Compute VALUE if requested
        value_batch = None
        if RequestTypes.VALUE in reqs:
            pv_array = pv_batch_fn(
                dom_payment_times_batch, dom_start_times_batch, dom_end_times_batch,
                dom_alphas_batch, dom_spreads_batch, dom_notionals_batch,
                dom_principal_batch, dom_leg_sign_batch, dom_notional_exchange_batch,
                dom_effective_time_batch, dom_maturity_time_batch,
                for_payment_times_batch, for_start_times_batch, for_end_times_batch,
                for_alphas_batch, for_spreads_batch, for_notionals_batch,
                for_principal_batch, for_leg_sign_batch, for_notional_exchange_batch,
                for_effective_time_batch, for_maturity_time_batch
            )
            value_batch = [float(pv) for pv in pv_array]

        # Compute FX01 if requested (requires foreign leg PV)
        fx_delta_batch = None
        if RequestTypes.FX01 in reqs:
            # Create vectorized function for foreign leg PV only
            # Foreign leg uses XCCY curve for discounting and foreign OIS for forward rates
            for_pv_batch_fn = vmap(
                lambda for_pmt, for_start, for_end, for_alpha, for_spr, for_not,
                       for_prin, for_sign, for_notex, for_eff, for_mat:
                    self._float_leg_jax(
                        dfs=xccy_dfs, times=xccy_times,
                        disc_interp_type=xccy_curve._interp_type,
                        idx_interp_type=foreign_model._interp_type,
                        payment_times=for_pmt,
                        start_times=for_start, end_times=for_end,
                        pay_alphas=for_alpha, spreads=for_spr,
                        notionals=for_not, principal=for_prin,
                        leg_sign=for_sign, value_time=value_time,
                        first_fixing_rate=0.0, override_first=False,
                        idx_times=for_times, idx_dfs=for_dfs,
                        notional_exchange=for_notex,
                        notional_exchange_amount=for_not[0],
                        effective_time=for_eff,
                        maturity_time=for_mat
                    ),
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # All foreign leg params batched
            )

            # Compute foreign PV for each swap
            for_pv_array = for_pv_batch_fn(
                for_payment_times_batch, for_start_times_batch, for_end_times_batch,
                for_alphas_batch, for_spreads_batch, for_notionals_batch,
                for_principal_batch, for_leg_sign_batch, for_notional_exchange_batch,
                for_effective_time_batch, for_maturity_time_batch
            )

            # Calculate FX01 for each swap: for_pv * spot_fx * 0.01
            fx_delta_batch = []
            for i, for_pv in enumerate(for_pv_array):
                fx01_sensitivity = float(for_pv) * spot_fx * 0.01
                fx_delta_obj = FXDelta(
                    sensitivity=fx01_sensitivity,
                    spot_fx=spot_fx,
                    currency=derivatives[i]._domestic_currency,
                    domestic_currency=derivatives[i]._domestic_currency,
                    foreign_currency=derivatives[i]._foreign_currency
                )
                fx_delta_batch.append(fx_delta_obj)

        # Compute DELTA if requested
        delta_batch = None
        if RequestTypes.DELTA in reqs:
            from cavour.utils.helpers import to_tenor
            from cavour.requests.results import Delta, Risk

            # Extract Jacobians (shared across batch)
            # Check if curves were built with AD (have stored Jacobians)
            if hasattr(domestic_model, '_jac') and domestic_model._jac is not None:
                dom_jac = domestic_model._jac
            else:
                raise LibError(
                    f"Domestic curve {dom_idx_name} must be built with use_ad=True for batched DELTA. "
                    "Rebuild curve with: model.build_curve(..., use_ad=True)"
                )

            if hasattr(foreign_model, '_jac') and foreign_model._jac is not None:
                for_jac = foreign_model._jac
            else:
                raise LibError(
                    f"Foreign curve {for_idx_name} must be built with use_ad=True for batched DELTA. "
                    "Rebuild curve with: model.build_curve(..., use_ad=True)"
                )

            if hasattr(xccy_curve, '_jac_basis') and xccy_curve._jac_basis is not None:
                # Skip first row if curve has prepended t=0 (to match original DFs)
                xccy_jac_basis = xccy_curve._jac_basis[1:, :] if xccy_times[0] < 1e-6 else xccy_curve._jac_basis
            else:
                raise LibError(
                    f"XCCY curve {xccy_curve_name} must be built with use_ad=True for batched DELTA. "
                    "Rebuild curve with: model.build_xccy_curve(..., use_ad=True)"
                )

            # Create vectorized DELTA function
            # in_axes: None = shared (curves, Jacobians), 0 = batched (swap params)
            delta_batch_fn = vmap(
                lambda dom_pmt, dom_start, dom_end, dom_alpha, dom_spr, dom_not,
                       dom_prin, dom_sign, dom_notex, dom_eff, dom_mat,
                       for_pmt, for_start, for_end, for_alpha, for_spr, for_not,
                       for_prin, for_sign, for_notex, for_eff, for_mat:
                    self._xccy_delta_pure(
                        dom_dfs, dom_times, domestic_model._interp_type,
                        for_dfs, for_times, foreign_model._interp_type,
                        xccy_dfs, xccy_times, xccy_curve._interp_type,
                        dom_jac, for_jac, xccy_jac_basis,
                        dom_pmt, dom_start, dom_end, dom_alpha, dom_spr, dom_not,
                        dom_prin, dom_sign, dom_notex, dom_eff, dom_mat,
                        for_pmt, for_start, for_end, for_alpha, for_spr, for_not,
                        for_prin, for_sign, for_notex, for_eff, for_mat,
                        value_time, spot_fx
                    ),
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Domestic leg batched
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)   # Foreign leg batched
            )

            # Compute batched DELTA
            deltas_tuple = delta_batch_fn(
                dom_payment_times_batch, dom_start_times_batch, dom_end_times_batch,
                dom_alphas_batch, dom_spreads_batch, dom_notionals_batch,
                dom_principal_batch, dom_leg_sign_batch, dom_notional_exchange_batch,
                dom_effective_time_batch, dom_maturity_time_batch,
                for_payment_times_batch, for_start_times_batch, for_end_times_batch,
                for_alphas_batch, for_spreads_batch, for_notionals_batch,
                for_principal_batch, for_leg_sign_batch, for_notional_exchange_batch,
                for_effective_time_batch, for_maturity_time_batch
            )

            # Unpack deltas (returned as tuple of 3 arrays)
            delta_dom_batch = deltas_tuple[0]  # [batch_size, n_dom_rates]
            delta_for_batch = deltas_tuple[1]  # [batch_size, n_for_rates]
            delta_basis_batch = deltas_tuple[2]  # [batch_size, n_basis]

            # Convert to tenor strings
            dom_tenors = to_tenor(domestic_model.swap_times)
            for_tenors = to_tenor(foreign_model.swap_times)
            basis_tenors = to_tenor(xccy_curve.swap_times)

            # Package into Delta/Risk objects for each swap
            delta_batch = []
            for i in range(batch_size):
                # Create Delta objects for each curve
                delta_domestic = Delta(
                    risk_ladder=[float(x) for x in delta_dom_batch[i]],
                    tenors=dom_tenors,
                    currency=derivatives[i]._domestic_currency,
                    curve_type=derivatives[i]._domestic_floating_index,
                )

                delta_foreign = Delta(
                    risk_ladder=[float(x) for x in delta_for_batch[i]],
                    tenors=for_tenors,
                    currency=derivatives[i]._domestic_currency,
                    curve_type=derivatives[i]._foreign_floating_index,
                )

                delta_basis = Delta(
                    risk_ladder=[float(x) for x in delta_basis_batch[i]],
                    tenors=basis_tenors,
                    currency=derivatives[i]._domestic_currency,
                    curve_type=CurveTypes.USD_GBP_BASIS,  # TODO: Derive from currencies
                )

                # Package into Risk object
                risk = Risk([delta_domestic, delta_foreign, delta_basis])
                delta_batch.append(risk)

        # Compute GAMMA if requested
        gamma_batch = None
        if RequestTypes.GAMMA in reqs:
            from cavour.utils.helpers import to_tenor
            from cavour.requests.results import Gamma, Risk

            # Extract Hessians (shared across batch)
            # Check if curves were built with compute_gamma=True (have stored Hessians)
            if hasattr(domestic_model, '_hess') and domestic_model._hess is not None:
                dom_hess = domestic_model._hess
            else:
                raise LibError(
                    f"Domestic curve {dom_idx_name} must be built with compute_gamma=True for batched GAMMA. "
                    "Rebuild curve with: model.build_curve(..., use_ad=True, compute_gamma=True)"
                )

            if hasattr(foreign_model, '_hess') and foreign_model._hess is not None:
                for_hess = foreign_model._hess
            else:
                raise LibError(
                    f"Foreign curve {for_idx_name} must be built with compute_gamma=True for batched GAMMA. "
                    "Rebuild curve with: model.build_curve(..., use_ad=True, compute_gamma=True)"
                )

            if hasattr(xccy_curve, '_hess_basis') and xccy_curve._hess_basis is not None:
                # Skip first row if curve has prepended t=0 (same as Jacobian)
                if xccy_times[0] < 1e-6:
                    if xccy_curve._hess_basis.ndim == 2:
                        xccy_hess_basis = xccy_curve._hess_basis[1:, :]  # Diagonal
                    else:
                        xccy_hess_basis = xccy_curve._hess_basis[1:, :, :]  # Full
                else:
                    xccy_hess_basis = xccy_curve._hess_basis
            else:
                raise LibError(
                    f"XCCY curve {xccy_curve_name} must be built with compute_gamma=True for batched GAMMA. "
                    "Rebuild curve with: model.build_xccy_curve(..., use_ad=True, compute_gamma=True)"
                )

            # Also need Jacobians for GAMMA computation
            if not hasattr(domestic_model, '_jac') or domestic_model._jac is None:
                raise LibError(
                    f"Domestic curve {dom_idx_name} must be built with use_ad=True for GAMMA. "
                    "Rebuild curve with: model.build_curve(..., use_ad=True, compute_gamma=True)"
                )
            if not hasattr(foreign_model, '_jac') or foreign_model._jac is None:
                raise LibError(
                    f"Foreign curve {for_idx_name} must be built with use_ad=True for GAMMA. "
                    "Rebuild curve with: model.build_curve(..., use_ad=True, compute_gamma=True)"
                )

            dom_jac = domestic_model._jac
            for_jac = foreign_model._jac
            xccy_jac_basis = xccy_curve._jac_basis[1:, :] if xccy_times[0] < 1e-6 else xccy_curve._jac_basis

            # Create vectorized GAMMA function
            # in_axes: None = shared (curves, Jacobians, Hessians), 0 = batched (swap params)
            gamma_batch_fn = vmap(
                lambda dom_pmt, dom_start, dom_end, dom_alpha, dom_spr, dom_not,
                       dom_prin, dom_sign, dom_notex, dom_eff, dom_mat,
                       for_pmt, for_start, for_end, for_alpha, for_spr, for_not,
                       for_prin, for_sign, for_notex, for_eff, for_mat:
                    self._xccy_gamma_pure(
                        dom_dfs, dom_times, domestic_model._interp_type,
                        for_dfs, for_times, foreign_model._interp_type,
                        xccy_dfs, xccy_times, xccy_curve._interp_type,
                        dom_jac, for_jac, xccy_jac_basis,
                        dom_hess, for_hess, xccy_hess_basis,
                        dom_pmt, dom_start, dom_end, dom_alpha, dom_spr, dom_not,
                        dom_prin, dom_sign, dom_notex, dom_eff, dom_mat,
                        for_pmt, for_start, for_end, for_alpha, for_spr, for_not,
                        for_prin, for_sign, for_notex, for_eff, for_mat,
                        value_time, spot_fx
                    ),
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Domestic leg batched
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)   # Foreign leg batched
            )

            # Compute batched GAMMA
            gammas_tuple = gamma_batch_fn(
                dom_payment_times_batch, dom_start_times_batch, dom_end_times_batch,
                dom_alphas_batch, dom_spreads_batch, dom_notionals_batch,
                dom_principal_batch, dom_leg_sign_batch, dom_notional_exchange_batch,
                dom_effective_time_batch, dom_maturity_time_batch,
                for_payment_times_batch, for_start_times_batch, for_end_times_batch,
                for_alphas_batch, for_spreads_batch, for_notionals_batch,
                for_principal_batch, for_leg_sign_batch, for_notional_exchange_batch,
                for_effective_time_batch, for_maturity_time_batch
            )

            # Unpack gammas (returned as tuple of 3 matrices)
            gamma_dom_batch = gammas_tuple[0]  # [batch_size, n_dom_rates, n_dom_rates]
            gamma_for_batch = gammas_tuple[1]  # [batch_size, n_for_rates, n_for_rates]
            gamma_basis_batch = gammas_tuple[2]  # [batch_size, n_basis, n_basis]

            # Convert to tenor strings
            dom_tenors = to_tenor(domestic_model.swap_times)
            for_tenors = to_tenor(foreign_model.swap_times)
            basis_tenors = to_tenor(xccy_curve.swap_times)

            # Package into Gamma/Risk objects for each swap
            gamma_batch = []
            for i in range(batch_size):
                # Create Gamma objects for each curve
                gamma_domestic = Gamma(
                    risk_ladder=gamma_dom_batch[i],  # Already numpy array
                    tenors=dom_tenors,
                    currency=derivatives[i]._domestic_currency,
                    curve_type=derivatives[i]._domestic_floating_index,
                )

                gamma_foreign = Gamma(
                    risk_ladder=gamma_for_batch[i],
                    tenors=for_tenors,
                    currency=derivatives[i]._domestic_currency,
                    curve_type=derivatives[i]._foreign_floating_index,
                )

                gamma_basis = Gamma(
                    risk_ladder=gamma_basis_batch[i],
                    tenors=basis_tenors,
                    currency=derivatives[i]._domestic_currency,
                    curve_type=CurveTypes.USD_GBP_BASIS,  # TODO: Derive from currencies
                )

                # Package into Risk object (with cross-gammas=None for now)
                risk = Risk([gamma_domestic, gamma_foreign, gamma_basis], cross_gammas=None)
                gamma_batch.append(risk)

        # Package results
        results = []
        for i in range(batch_size):
            value_obj = Valuation(amount=value_batch[i], currency=derivatives[i]._domestic_currency) if value_batch else None
            delta_obj = delta_batch[i] if delta_batch else None
            gamma_obj = gamma_batch[i] if gamma_batch else None
            fx_delta_obj = fx_delta_batch[i] if fx_delta_batch else None
            result = AnalyticsResult(value=value_obj, risk=delta_obj, gamma=gamma_obj, fx_delta=fx_delta_obj)
            results.append(result)

        return results

    def _compute_batch_ois(self, derivatives, reqs):
        """Internal method for batched OIS swap computation (natural currency collateral).

        Handles VALUE and DELTA requests for OIS swaps using JAX vectorization.
        Simpler than XCCY: fixed leg + floating leg, single curve, no FX conversion.

        Expected performance: 3-6x speedup for batches of 10+ swaps.
        """
        from jax import vmap
        import jax.numpy as jnp
        import numpy as np
        from cavour.utils.helpers import times_from_dates, to_tenor
        from cavour.requests.results import AnalyticsResult, Valuation, Delta, Risk

        # Validate all derivatives are OIS swaps
        for i, deriv in enumerate(derivatives):
            if deriv.derivative_type != InstrumentTypes.OIS_SWAP:
                raise LibError(f"All derivatives must be OIS swaps. derivatives[{i}] is {deriv.derivative_type}")

        # Validate all swaps use same OIS curve (same floating index)
        first = derivatives[0]
        ois_idx_name = first._floating_index.name

        for i, deriv in enumerate(derivatives):
            if deriv._floating_index.name != ois_idx_name:
                raise LibError(
                    f"All OIS swaps must use same floating index. "
                    f"derivatives[{i}] uses {deriv._floating_index.name}, expected {ois_idx_name}"
                )

        # Get OIS curve (shared across batch)
        ois_model = getattr(self.model.curves, ois_idx_name)

        # Extract curve data once (shared across all swaps)
        ois_times = jnp.array(ois_model._times)
        ois_dfs = jnp.array(ois_model._dfs)

        # Get valuation time
        value_dt = ois_model._value_dt
        value_time = times_from_dates([value_dt], value_dt, ois_model._dc_type)[0]

        batch_size = len(derivatives)

        # Extract and pad fixed leg parameters
        max_fixed_cashflows = max(len(d._fixed_leg._payment_dts) for d in derivatives)

        fixed_payment_times_batch = jnp.zeros((batch_size, max_fixed_cashflows))
        fixed_payments_batch = jnp.zeros((batch_size, max_fixed_cashflows))
        fixed_principal_batch = jnp.zeros(batch_size)
        fixed_leg_sign_batch = jnp.zeros(batch_size)

        for i, deriv in enumerate(derivatives):
            n_fixed = len(deriv._fixed_leg._payment_dts)

            # Payment times
            times_fixed = jnp.array(times_from_dates(
                deriv._fixed_leg._payment_dts,
                value_dt,
                deriv._fixed_leg._dc_type
            ))
            fixed_payment_times_batch = fixed_payment_times_batch.at[i, :n_fixed].set(times_fixed)

            # Payments
            payments_fixed = jnp.array(deriv._fixed_leg._payments)
            fixed_payments_batch = fixed_payments_batch.at[i, :n_fixed].set(payments_fixed)

            # Principal and leg sign
            fixed_principal_batch = fixed_principal_batch.at[i].set(deriv._fixed_leg._principal)
            leg_sign = +1.0 if deriv._fixed_leg._leg_type == SwapTypes.RECEIVE else -1.0
            fixed_leg_sign_batch = fixed_leg_sign_batch.at[i].set(leg_sign)

        # Extract and pad float leg parameters
        max_float_cashflows = max(len(d._float_leg._payment_dts) for d in derivatives)

        float_payment_times_batch = jnp.zeros((batch_size, max_float_cashflows))
        float_start_times_batch = jnp.zeros((batch_size, max_float_cashflows))
        float_end_times_batch = jnp.zeros((batch_size, max_float_cashflows))
        float_alphas_batch = jnp.zeros((batch_size, max_float_cashflows))
        float_spreads_batch = jnp.zeros((batch_size, max_float_cashflows))
        float_notionals_batch = jnp.zeros((batch_size, max_float_cashflows))
        float_principal_batch = jnp.zeros(batch_size)
        float_leg_sign_batch = jnp.zeros(batch_size)

        for i, deriv in enumerate(derivatives):
            n_float = len(deriv._float_leg._payment_dts)

            # Payment times
            times_float = jnp.array(times_from_dates(
                deriv._float_leg._payment_dts,
                value_dt,
                deriv._float_leg._dc_type
            ))
            float_payment_times_batch = float_payment_times_batch.at[i, :n_float].set(times_float)

            # Start times (for forward rate calculation)
            start_times_float = jnp.array(times_from_dates(
                deriv._float_leg._start_accrued_dts,
                value_dt,
                deriv._float_leg._dc_type
            ))
            float_start_times_batch = float_start_times_batch.at[i, :n_float].set(start_times_float)

            # End times (for forward rate calculation)
            end_times_float = jnp.array(times_from_dates(
                deriv._float_leg._end_accrued_dts,
                value_dt,
                deriv._float_leg._dc_type
            ))
            float_end_times_batch = float_end_times_batch.at[i, :n_float].set(end_times_float)

            # Year fractions (alphas)
            alphas_float = jnp.array(deriv._float_leg._year_fracs)
            float_alphas_batch = float_alphas_batch.at[i, :n_float].set(alphas_float)

            # Spreads (broadcast scalar to array)
            spreads_float = np.full(n_float, deriv._float_leg._spread)
            float_spreads_batch = float_spreads_batch.at[i, :n_float].set(spreads_float)

            # Notionals (use _notional_array if available, otherwise broadcast _notional)
            if hasattr(deriv._float_leg, '_notional_array') and deriv._float_leg._notional_array:
                notionals_float = np.array(deriv._float_leg._notional_array)
            else:
                notionals_float = np.full(n_float, deriv._float_leg._notional)
            float_notionals_batch = float_notionals_batch.at[i, :n_float].set(notionals_float)

            # Principal and leg sign
            float_principal_batch = float_principal_batch.at[i].set(deriv._float_leg._principal)
            leg_sign = +1.0 if deriv._float_leg._leg_type == SwapTypes.RECEIVE else -1.0
            float_leg_sign_batch = float_leg_sign_batch.at[i].set(leg_sign)

        # Compute VALUE if requested
        value_batch = None
        if RequestTypes.VALUE in reqs:
            # Create vectorized PV function
            pv_batch_fn = vmap(
                lambda fixed_pmt, fixed_pay, fixed_prin, fixed_sign,
                       float_pmt, float_start, float_end, float_alpha, float_spr, float_not, float_prin, float_sign:
                    self._ois_pv_pure(
                        ois_dfs, ois_times, ois_model._interp_type,
                        fixed_pmt, fixed_pay, fixed_prin, fixed_sign,
                        float_pmt, float_start, float_end, float_alpha, float_spr, float_not, float_prin, float_sign,
                        value_time
                    ),
                in_axes=(0, 0, 0, 0,  # Fixed leg batched
                         0, 0, 0, 0, 0, 0, 0, 0)  # Float leg batched
            )

            # Compute batched PV
            pv_array = pv_batch_fn(
                fixed_payment_times_batch, fixed_payments_batch, fixed_principal_batch, fixed_leg_sign_batch,
                float_payment_times_batch, float_start_times_batch, float_end_times_batch,
                float_alphas_batch, float_spreads_batch, float_notionals_batch,
                float_principal_batch, float_leg_sign_batch
            )

            # Convert to Python list (ensure scalars)
            # pv_array may have extra dimensions, so squeeze and flatten
            pv_array_flat = jnp.atleast_1d(jnp.squeeze(pv_array))
            value_batch = [float(pv) for pv in pv_array_flat]

        # Compute DELTA if requested
        delta_batch = None
        if RequestTypes.DELTA in reqs:
            # Extract Jacobian (shared across batch)
            if hasattr(ois_model, '_jac') and ois_model._jac is not None:
                ois_jac = ois_model._jac
            else:
                raise LibError(
                    f"OIS curve {ois_idx_name} must be built with use_ad=True for batched DELTA. "
                    "Rebuild curve with: model.build_curve(..., use_ad=True)"
                )

            # Create vectorized DELTA function
            delta_batch_fn = vmap(
                lambda fixed_pmt, fixed_pay, fixed_prin, fixed_sign,
                       float_pmt, float_start, float_end, float_alpha, float_spr, float_not, float_prin, float_sign:
                    self._ois_delta_pure(
                        ois_dfs, ois_times, ois_model._interp_type, ois_jac,
                        fixed_pmt, fixed_pay, fixed_prin, fixed_sign,
                        float_pmt, float_start, float_end, float_alpha, float_spr, float_not, float_prin, float_sign,
                        value_time
                    ),
                in_axes=(0, 0, 0, 0,  # Fixed leg batched
                         0, 0, 0, 0, 0, 0, 0, 0)  # Float leg batched
            )

            # Compute batched DELTA
            delta_array = delta_batch_fn(
                fixed_payment_times_batch, fixed_payments_batch, fixed_principal_batch, fixed_leg_sign_batch,
                float_payment_times_batch, float_start_times_batch, float_end_times_batch,
                float_alphas_batch, float_spreads_batch, float_notionals_batch,
                float_principal_batch, float_leg_sign_batch
            )

            # Convert to tenor strings
            ois_tenors = to_tenor(ois_model.swap_times)

            # Package into Delta objects for each swap
            # Note: OIS sequential code returns a single Delta object (not wrapped in Risk)
            # This is different from XCCY which returns Risk with multiple Delta objects
            delta_batch = []
            for i in range(batch_size):
                delta_ois = Delta(
                    risk_ladder=[float(x) for x in delta_array[i]],
                    tenors=ois_tenors,
                    currency=derivatives[i]._currency,
                    curve_type=derivatives[i]._floating_index,
                )
                delta_batch.append(delta_ois)

        # Package results
        # Compute GAMMA if requested
        gamma_batch = None
        if RequestTypes.GAMMA in reqs:
            from cavour.utils.helpers import to_tenor
            from cavour.requests.results import Gamma
            from jax import hessian

            # Extract Hessian (shared across batch)
            # IMPORTANT: Must recompute full 3D Hessian to match sequential behavior
            # The stored _hess is 2D diagonal when hessian_bandwidth=0, but sequential
            # code uses _cached_curve which always computes full 3D Hessian
            # Also need Jacobian for GAMMA computation (should already be available from DELTA)
            if not hasattr(ois_model, '_jac') or ois_model._jac is None:
                raise LibError(
                    f"OIS curve {ois_curve_name} must be built with use_ad=True for GAMMA. "
                    "Rebuild curve with: model.build_curve(..., use_ad=True, compute_gamma=True)"
                )

            ois_jac = ois_model._jac

            # Recompute full 3D Hessian (matching _cached_curve behavior)
            # IMPORTANT: build_curve_ad returns DFs including prepended t=0 point
            # We need to strip the first row to match stored _jac shape
            swap_rates_jax = jnp.array(ois_model.swap_rates)
            swap_times_list = ois_model.swap_times
            year_fracs_list = ois_model.year_fracs
            start_times_list = ois_model.start_times  # NEW: For forward-starting instruments

            def build_dfs_original(r):
                _, dfs_out = self.build_curve_ad(r, swap_times_list, year_fracs_list, start_times_list)
                return dfs_out

            hess_full = hessian(build_dfs_original)(swap_rates_jax)

            # Strip first row if curve has prepended t=0 (matching stored _jac shape)
            if ois_times[0] < 1e-6:
                ois_hess = hess_full[1:, :, :]  # Remove t=0 row
            else:
                ois_hess = hess_full

            # Create vectorized GAMMA function
            # in_axes: None = shared (curves, Jacobian, Hessian), 0 = batched (swap params)
            gamma_batch_fn = vmap(
                lambda fixed_pmt, fixed_pay, fixed_prin, fixed_sign,
                       float_pmt, float_start, float_end, float_alpha, float_spr, float_not, float_prin, float_sign:
                    self._ois_gamma_pure(
                        ois_dfs, ois_times, ois_model._interp_type, ois_jac, ois_hess,
                        fixed_pmt, fixed_pay, fixed_prin, fixed_sign,
                        float_pmt, float_start, float_end, float_alpha, float_spr, float_not, float_prin, float_sign,
                        value_time
                    ),
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            )

            # Compute batched GAMMA
            gamma_array = gamma_batch_fn(
                fixed_payment_times_batch, fixed_payments_batch, fixed_principal_batch, fixed_leg_sign_batch,
                float_payment_times_batch, float_start_times_batch, float_end_times_batch,
                float_alphas_batch, float_spreads_batch, float_notionals_batch,
                float_principal_batch, float_leg_sign_batch
            )

            # Convert to tenor strings
            ois_tenors = to_tenor(ois_model.swap_times)

            # Package into Gamma objects for each swap
            # Note: OIS returns single Gamma object (not wrapped in Risk like XCCY)
            gamma_batch = []
            for i in range(batch_size):
                gamma_ois = Gamma(
                    risk_ladder=gamma_array[i],  # Already numpy array
                    tenors=ois_tenors,
                    currency=derivatives[i]._currency,
                    curve_type=derivatives[i]._floating_index,
                )
                gamma_batch.append(gamma_ois)

        # Package results
        results = []
        for i in range(batch_size):
            value_obj = None
            if value_batch is not None:
                value_obj = Valuation(
                    amount=value_batch[i],
                    currency=derivatives[i]._currency
                )

            delta_obj = delta_batch[i] if delta_batch is not None else None
            gamma_obj = gamma_batch[i] if gamma_batch is not None else None

            result = AnalyticsResult(value=value_obj, risk=delta_obj, gamma=gamma_obj)
            results.append(result)

        return results

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
            ois_model._interp_type,
            ois_model.start_times  # NEW: For forward-starting instruments
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

        # FX01 calculation (FX sensitivity)
        fx_delta = None
        if RequestTypes.FX01 in reqs:
            # For cross-currency collateral, PV in collateral currency is:
            # PV_collateral = PV_swap / spot_fx
            # The derivative with respect to spot_fx is:
            # dPV/d(spot_fx) = -PV_swap / spot_fx^2
            # For 1% move: FX01 = -PV_swap / spot_fx^2 * spot_fx * 0.01
            #              = -PV_swap / spot_fx * 0.01
            #              = -PV_collateral * 0.01
            #
            # Negative sign: when spot_fx increases (swap currency weakens),
            # the collateral currency value decreases
            fx01_sensitivity = -total_pv_collateral_ccy * 0.01

            fx_delta = FXDelta(
                sensitivity=fx01_sensitivity,
                spot_fx=spot_fx,
                currency=collateral_ccy,
                domestic_currency=collateral_ccy,  # Collateral is the "domestic" for this calculation
                foreign_currency=derivative._currency  # Swap currency is the "foreign"
            )

        # Cashflows extraction (placeholder for future implementation)
        cashflows = None
        if RequestTypes.CASHFLOWS in reqs:
            # TODO: Extract cashflow data from fixed and floating legs
            cashflows = Cashflows([], derivative._currency)

        return AnalyticsResult(value=value, risk=delta, gamma=gamma, cashflows=cashflows, fx_delta=fx_delta)

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

        # Use SensitivityEngine for DELTA and GAMMA computation (centralized implementation)
        from cavour.market.sensitivity import SensitivityEngine

        # Create curve type enum based on currency
        curve_type_map = {
            CurrencyTypes.GBP: CurveTypes.GBP_OIS_SONIA,
            CurrencyTypes.USD: CurveTypes.USD_OIS_SOFR,
            CurrencyTypes.EUR: CurveTypes.EUR_OIS_ESTR,
        }
        curve_type = curve_type_map.get(derivative._currency, CurveTypes.GBP_OIS_SONIA)

        need_both = RequestTypes.DELTA in reqs and RequestTypes.GAMMA in reqs
        if need_both:
            # Check GAMMA precondition
            if hess_curve is None:
                raise LibError("GAMMA requested but curve was not built with compute_gamma=True")

            # Compute both DELTA and GAMMA efficiently (shares gradient computation)
            delta, gamma = SensitivityEngine.compute_delta_gamma(
                pv_fn=pv_fn,
                dfs=dfs,
                jac=jac,
                hess_curve=hess_curve,
                swap_times=ir_model.swap_times,
                currency=derivative._currency,
                curve_type=curve_type
            )
        else:
            # Compute only what's requested
            if RequestTypes.DELTA in reqs:
                delta = SensitivityEngine.compute_delta(
                    pv_fn=pv_fn,
                    dfs=dfs,
                    jac=jac,
                    swap_times=ir_model.swap_times,
                    currency=derivative._currency,
                    curve_type=curve_type
                )

            if RequestTypes.GAMMA in reqs:
                # Check GAMMA precondition
                if hess_curve is None:
                    raise LibError("GAMMA requested but curve was not built with compute_gamma=True")

                gamma = SensitivityEngine.compute_gamma(
                    pv_fn=pv_fn,
                    dfs=dfs,
                    jac=jac,
                    hess_curve=hess_curve,
                    grad_dfs=None,
                    swap_times=ir_model.swap_times,
                    currency=derivative._currency,
                    curve_type=curve_type
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
                # Single curve case - use SensitivityEngine for centralized computation
                from cavour.market.sensitivity import SensitivityEngine

                curve_type_map = {
                    CurrencyTypes.GBP: CurveTypes.GBP_OIS_SONIA,
                    CurrencyTypes.USD: CurveTypes.USD_OIS_SOFR,
                    CurrencyTypes.EUR: CurveTypes.EUR_OIS_ESTR,
                }
                curve_type = curve_type_map.get(derivative._currency, CurveTypes.GBP_OIS_SONIA)

                need_both = RequestTypes.DELTA in reqs and RequestTypes.GAMMA in reqs
                if need_both:
                    # Check GAMMA precondition
                    if disc_hess is None:
                        raise LibError("GAMMA requested but curve was not built with compute_gamma=True")

                    # Compute both DELTA and GAMMA efficiently (shares gradient computation)
                    delta, gamma = SensitivityEngine.compute_delta_gamma(
                        pv_fn=pv_fn_combined,
                        dfs=disc_dfs,
                        jac=disc_jac,
                        hess_curve=disc_hess,
                        swap_times=discount_model.swap_times,
                        currency=derivative._currency,
                        curve_type=curve_type
                    )
                else:
                    # Compute only what's requested
                    if RequestTypes.DELTA in reqs:
                        delta = SensitivityEngine.compute_delta(
                            pv_fn=pv_fn_combined,
                            dfs=disc_dfs,
                            jac=disc_jac,
                            swap_times=discount_model.swap_times,
                            currency=derivative._currency,
                            curve_type=curve_type
                        )

                    if RequestTypes.GAMMA in reqs:
                        # Check GAMMA precondition
                        if disc_hess is None:
                            raise LibError("GAMMA requested but curve was not built with compute_gamma=True")

                        gamma = SensitivityEngine.compute_gamma(
                            pv_fn=pv_fn_combined,
                            dfs=disc_dfs,
                            jac=disc_jac,
                            hess_curve=disc_hess,
                            grad_dfs=None,
                            swap_times=discount_model.swap_times,
                            currency=derivative._currency,
                            curve_type=curve_type
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
        xccy_curve_name = f"{domestic_code}_{foreign_code}_BASIS"  # Match CurveTypes enum pattern

        try:
            xccy_curve = getattr(self.model.curves, xccy_curve_name)
            spot_fx = xccy_curve._spot_fx
        except AttributeError:
            raise LibError(f"XCCY curve {xccy_curve_name} not found in model. "
                         f"Available curves: {[attr for attr in dir(self.model.curves) if not attr.startswith('_')]}")

        # Get domestic OIS curve arrays directly from stored curve
        # IMPORTANT: Use curve's existing _times and _dfs (from original bootstrap)
        # DO NOT re-bootstrap via build_curve_ad() as it creates numerical differences
        dom_times = jnp.array(domestic_model._times)
        dom_dfs = jnp.array(domestic_model._dfs)

        # Prepend t=0 point if not present (needed for interpolation at value date)
        if dom_times[0] > 1e-7:
            dom_times = jnp.concatenate([jnp.array([1e-8]), dom_times])
            dom_dfs = jnp.concatenate([jnp.array([1.0]), dom_dfs])

        # Get foreign OIS curve arrays directly from stored curve
        # IMPORTANT: Use curve's existing _times and _dfs (from original bootstrap)
        # DO NOT re-bootstrap via build_curve_ad() as it creates numerical differences
        for_times = jnp.array(foreign_model._times)
        for_dfs = jnp.array(foreign_model._dfs)

        # Prepend t=0 point if not present (needed for interpolation at value date)
        if for_times[0] > 1e-7:
            for_times = jnp.concatenate([jnp.array([1e-8]), for_times])
            for_dfs = jnp.concatenate([jnp.array([1.0]), for_dfs])

        # Get XCCY curve arrays
        # Note: XCCY curve times are in ACT_365F, but we'll use them to interpolate
        # foreign leg payment times in ACT_360. This creates a small time mismatch
        # (similar to how .value() method handles it)
        xccy_times = jnp.array(xccy_curve._times)
        xccy_dfs = jnp.array(xccy_curve._dfs)

        # Prepare leg parameters for JAX computation
        dc_type = derivative._domestic_leg._dc_type
        value_time = times_from_dates(self.model.value_dt, self.model.value_dt, dc_type)

        # Detect leg types to route appropriately (following VALUE section pattern)
        from cavour.trades.rates.swap_fixed_leg import SwapFixedLeg
        is_domestic_fixed = isinstance(derivative._domestic_leg, SwapFixedLeg)
        is_foreign_fixed = isinstance(derivative._foreign_leg, SwapFixedLeg)

        # Domestic leg parameters (conditional based on leg type)
        if is_domestic_fixed:
            # Fixed leg: extract predetermined payments
            dom_payment_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                           for dt in derivative._domestic_leg._payment_dts])
            dom_payments = jnp.array(derivative._domestic_leg._payments)
            dom_principal = derivative._domestic_leg._principal
            dom_notional = derivative._domestic_leg._notional
            dom_leg_sign = +1.0 if derivative._domestic_leg._leg_type == SwapTypes.RECEIVE else -1.0

            # Fixed legs always have notional exchanges in XCCY
            dom_notional_exchange = True
            dom_notional_exchange_amount = derivative._domestic_leg._notional
        else:
            # Floating leg: extract forward rate parameters
            dom_payment_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                           for dt in derivative._domestic_leg._payment_dts])
            dom_start_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                         for dt in derivative._domestic_leg._start_accrued_dts])
            dom_end_times = jnp.array([times_from_dates(dt, self.model.value_dt, dc_type)
                                       for dt in derivative._domestic_leg._end_accrued_dts])
            dom_alphas = jnp.array(derivative._domestic_leg._year_fracs)
            dom_spreads = jnp.full_like(dom_alphas, derivative._domestic_leg._spread)
            dom_notionals = jnp.array(getattr(derivative._domestic_leg, '_notional_array', None) or
                                      [derivative._domestic_leg._notional] * len(dom_alphas))
            dom_principal = derivative._domestic_leg._principal
            dom_leg_sign = +1.0 if derivative._domestic_leg._leg_type == SwapTypes.RECEIVE else -1.0

            dom_notional_exchange = getattr(derivative._domestic_leg, '_notional_exchange', True)
            dom_notional_exchange_amount = derivative._domestic_leg._notional

        # Foreign leg parameters (conditional based on leg type)
        # For discounting: ALWAYS use XCCY curve's day count (ACT_365F)
        xccy_dc_type = xccy_curve._dc_type  # ACT_365F for discounting

        if is_foreign_fixed:
            # Fixed leg: extract predetermined payments (only need discounting, no forward rates)
            for_payment_times = jnp.array([times_from_dates(dt, self.model.value_dt, xccy_dc_type)
                                           for dt in derivative._foreign_leg._payment_dts])
            for_payments = jnp.array(derivative._foreign_leg._payments)
            for_principal = derivative._foreign_leg._principal
            for_notional = derivative._foreign_leg._notional
            for_leg_sign = +1.0 if derivative._foreign_leg._leg_type == SwapTypes.RECEIVE else -1.0

            # Fixed legs always have notional exchanges in XCCY
            for_notional_exchange = True
            for_notional_exchange_amount = derivative._foreign_leg._notional
        else:
            # Floating leg: extract forward rate parameters
            # For forward rates: use foreign leg's day count (ACT_360) to match foreign OIS curve
            for_dc_type = derivative._foreign_leg._dc_type  # ACT_360 for forward rates

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
            for_notionals = jnp.array(getattr(derivative._foreign_leg, '_notional_array', None) or
                                      [derivative._foreign_leg._notional] * len(for_alphas))
            for_principal = derivative._foreign_leg._principal
            for_leg_sign = +1.0 if derivative._foreign_leg._leg_type == SwapTypes.RECEIVE else -1.0

            for_notional_exchange = getattr(derivative._foreign_leg, '_notional_exchange', True)
            for_notional_exchange_amount = derivative._foreign_leg._notional

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
            # Domestic leg PV (conditional based on leg type)
            if is_domestic_fixed:
                # Fixed leg: predetermined payments
                dom_pv = self._fixed_leg_jax(
                    dfs=dom_dfs,
                    times=dom_times,
                    disc_interp_type=domestic_model._interp_type,
                    payment_times=dom_payment_times,
                    payments=dom_payments,
                    principal=dom_principal,
                    leg_sign=dom_leg_sign,
                    value_time=value_time,
                    notional_exchange=dom_notional_exchange,
                    notional_exchange_amount=dom_notional_exchange_amount,
                    effective_time=dom_effective_time,
                    maturity_time=dom_maturity_time
                )
            else:
                # Floating leg: forward rate projection
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
                    notional_exchange=dom_notional_exchange,
                    notional_exchange_amount=dom_notional_exchange_amount,
                    effective_time=dom_effective_time,
                    maturity_time=dom_maturity_time
                )

            # Foreign leg PV (conditional based on leg type)
            if is_foreign_fixed:
                # Fixed leg: predetermined payments, XCCY discounting only
                for_pv = self._fixed_leg_jax(
                    dfs=xccy_dfs,
                    times=xccy_times,
                    disc_interp_type=xccy_curve._interp_type,
                    payment_times=for_payment_times,
                    payments=for_payments,
                    principal=for_principal,
                    leg_sign=for_leg_sign,
                    value_time=value_time,
                    notional_exchange=for_notional_exchange,
                    notional_exchange_amount=for_notional_exchange_amount,
                    effective_time=for_effective_time,
                    maturity_time=for_maturity_time
                )
            else:
                # Floating leg: dual curve (XCCY for discount, foreign OIS for index)
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
                    notional_exchange=for_notional_exchange,
                    notional_exchange_amount=for_notional_exchange_amount,
                    effective_time=for_effective_time,
                    maturity_time=for_maturity_time
                )

            # Convert to scalars and compute total PV
            # spot_fx is USD/GBP (domestic/foreign)
            # for_pv_scalar is in GBP, multiply by spot_fx to convert to USD
            dom_pv_scalar = float(jnp.squeeze(dom_pv))
            for_pv_scalar = float(jnp.squeeze(for_pv))
            total_pv = dom_pv_scalar + for_pv_scalar * spot_fx
            value = Valuation(amount=total_pv, currency=derivative._domestic_currency)

        # Define PV functions for gradient/hessian computation (used by both DELTA and GAMMA)
        # These functions compute PV as a function of different curve variables

        # Domestic leg PV as function of domestic DFs
        if is_domestic_fixed:
            # Fixed leg: predetermined payments, only discounting varies with DFs
            def pv_dom_fn(dom_dfs_var):
                return self._fixed_leg_jax(
                    dfs=dom_dfs_var, times=dom_times,
                    disc_interp_type=domestic_model._interp_type,
                    payment_times=dom_payment_times,
                    payments=dom_payments,
                    principal=dom_principal,
                    leg_sign=dom_leg_sign,
                    value_time=value_time,
                    notional_exchange=dom_notional_exchange,
                    notional_exchange_amount=dom_notional_exchange_amount,
                    effective_time=dom_effective_time,
                    maturity_time=dom_maturity_time
                )
        else:
            # Floating leg: forward rates and discounting vary with DFs
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
                    notional_exchange=dom_notional_exchange,
                    notional_exchange_amount=dom_notional_exchange_amount,
                    effective_time=dom_effective_time,
                    maturity_time=dom_maturity_time
                )

        # Foreign leg PV as function of foreign OIS DFs (for direct forward rate effect)
        if is_foreign_fixed:
            # Fixed leg: NO dependency on foreign OIS DFs (no forward rates)
            # Return constant PV (gradient will be zero) to maintain consistency
            def pv_for_fn(for_ois_dfs_var):
                # Fixed payments discounted at XCCY curve (independent of for_ois_dfs_var)
                return self._fixed_leg_jax(
                    dfs=xccy_dfs, times=xccy_times,  # XCCY curve for discounting (FIXED)
                    disc_interp_type=xccy_curve._interp_type,
                    payment_times=for_payment_times,
                    payments=for_payments,
                    principal=for_principal,
                    leg_sign=for_leg_sign,
                    value_time=value_time,
                    notional_exchange=for_notional_exchange,
                    notional_exchange_amount=for_notional_exchange_amount,
                    effective_time=for_effective_time,
                    maturity_time=for_maturity_time
                )
        else:
            # Floating leg: forward rates depend on foreign OIS DFs
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
                    notional_exchange=for_notional_exchange,
                    notional_exchange_amount=for_notional_exchange_amount,
                    effective_time=for_effective_time,
                    maturity_time=for_maturity_time
                )

        # Foreign leg PV as function of XCCY DFs (for basis delta and cross-gamma)
        if is_foreign_fixed:
            # Fixed leg: discounting depends on XCCY DFs (basis spread sensitivity)
            def pv_xccy_fn(xccy_dfs_var):
                return self._fixed_leg_jax(
                    dfs=xccy_dfs_var, times=xccy_times,  # XCCY curve for discounting (VARIABLE)
                    disc_interp_type=xccy_curve._interp_type,
                    payment_times=for_payment_times,
                    payments=for_payments,
                    principal=for_principal,
                    leg_sign=for_leg_sign,
                    value_time=value_time,
                    notional_exchange=for_notional_exchange,
                    notional_exchange_amount=for_notional_exchange_amount,
                    effective_time=for_effective_time,
                    maturity_time=for_maturity_time
                )
        else:
            # Floating leg: discounting and forward rates both depend on curves
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
                    notional_exchange=for_notional_exchange,
                    notional_exchange_amount=for_notional_exchange_amount,
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
            # For DELTA, we need Jacobian d(DFs)/d(rates)
            # Use stored Jacobian if available (ensures consistency with VALUE)
            # Otherwise fallback to _cached_curve() for backward compatibility

            # Domestic OIS: Check for stored Jacobian
            if hasattr(domestic_model, '_jac') and domestic_model._jac is not None:
                # Use stored Jacobian from curve construction (consistent with VALUE)
                jac_dom_original = domestic_model._jac
            else:
                # Fallback: Rebuild curve and compute Jacobian (legacy behavior)
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

            # Foreign OIS: Check for stored Jacobian
            if hasattr(foreign_model, '_jac') and foreign_model._jac is not None:
                # Use stored Jacobian from curve construction (consistent with VALUE)
                jac_for_original = foreign_model._jac
            else:
                # Fallback: Rebuild curve and compute Jacobian (legacy behavior)
                for_cache = self._cached_curve(
                    tuple(foreign_model.swap_times),
                    foreign_model.swap_rates,
                    foreign_model.swap_times,
                    foreign_model.year_fracs,
                    foreign_model._interp_type
                )
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

            # Convert foreign OIS delta to domestic currency per bp
            # Foreign leg PV is in GBP (foreign currency)
            # Delta w.r.t. foreign rates is d(PV_GBP)/d(rate)
            # Multiply by spot_fx (USD/GBP) to convert to d(PV_USD)/d(rate)
            # Rates are stored in DECIMAL (0.052 for 5.2%), Jacobian is d(DFs)/d(rate_decimal)
            # 1bp = 0.0001 in decimal units → multiply by 1e-4
            delta_for_rates = [float(x) * 1e-4 * spot_fx for x in delta_for_rates_raw]

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

                # Convert basis delta to domestic currency per bp
                # Foreign leg PV is in GBP (foreign currency)
                # Delta w.r.t. basis spreads is d(PV_GBP)/d(spread)
                # Multiply by spot_fx (USD/GBP) to convert to d(PV_USD)/d(spread)
                # Basis spreads are stored in DECIMAL (0.0030 for 30bp), Jacobian is d(DFs)/d(spread_decimal)
                # 1bp = 0.0001 in decimal units → multiply by 1e-4
                delta_basis_rates = [float(x) * 1e-4 * spot_fx for x in delta_basis_rates_raw]

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

            # Domestic OIS: Get Jacobian and Hessian (use stored if available)
            if hasattr(domestic_model, '_jac') and domestic_model._jac is not None:
                jac_dom_original = domestic_model._jac
            else:
                # Need to compute via _cached_curve()
                dom_cache = self._cached_curve(
                    tuple(domestic_model.swap_times),
                    domestic_model.swap_rates,
                    domestic_model.swap_times,
                    domestic_model.year_fracs,
                    domestic_model._interp_type
                )
                jac_dom_original = dom_cache["jac"][1:, :] if dom_times[0] < 1e-6 else dom_cache["jac"]

            grad_for_dfs_original = grad(lambda d: jnp.squeeze(pv_for_original_dfs(d)))(for_ois_dfs_original)

            # Foreign OIS: Get Jacobian (use stored if available)
            if hasattr(foreign_model, '_jac') and foreign_model._jac is not None:
                jac_for_original = foreign_model._jac
            else:
                # Need to compute via _cached_curve()
                for_cache = self._cached_curve(
                    tuple(foreign_model.swap_times),
                    foreign_model.swap_rates,
                    foreign_model.swap_times,
                    foreign_model.year_fracs,
                    foreign_model._interp_type
                )
                jac_for_original = for_cache["jac"][1:, :] if for_times[0] < 1e-6 else for_cache["jac"]

            grad_xccy_dfs_original = grad(lambda d: jnp.squeeze(pv_xccy_original_dfs(d)))(xccy_dfs_original)

            # Domestic OIS GAMMA - use SensitivityEngine for centralized computation
            from cavour.market.sensitivity import SensitivityEngine

            # Get Hessian of curve bootstrapping (d²DFs/d(rates)²)
            if hasattr(domestic_model, '_hess') and domestic_model._hess is not None:
                # Use stored Hessian from curve construction
                # NOTE: Stored Hessians already exclude prepended t=0 row (shape matches jac)
                hess_dom_curve = domestic_model._hess
            else:
                # Fallback: Get from _cached_curve() (may have been computed above)
                if 'dom_cache' not in locals():
                    dom_cache = self._cached_curve(
                        tuple(domestic_model.swap_times),
                        domestic_model.swap_rates,
                        domestic_model.swap_times,
                        domestic_model.year_fracs,
                        domestic_model._interp_type
                    )
                hess_dom_curve = dom_cache["hess"][1:, :, :] if dom_times[0] < 1e-6 else dom_cache["hess"]

            # Check GAMMA precondition
            if hess_dom_curve is None:
                raise LibError("GAMMA requested but domestic curve was not built with compute_gamma=True")

            # Compute GAMMA using SensitivityEngine (reuses pre-computed gradient)
            gamma_domestic_obj = SensitivityEngine.compute_gamma(
                pv_fn=pv_dom_original_dfs,
                dfs=dom_dfs_original,
                jac=jac_dom_original,
                hess_curve=hess_dom_curve,
                grad_dfs=grad_dom_dfs_original,
                swap_times=domestic_model.swap_times,
                currency=derivative._domestic_currency,
                curve_type=derivative._domestic_floating_index
            )
            gammas_dom = gamma_domestic_obj.risk_ladder

            # Foreign OIS GAMMA - use SensitivityEngine for centralized computation
            # Get Hessian of curve bootstrapping (d²DFs/d(rates)²)
            if hasattr(foreign_model, '_hess') and foreign_model._hess is not None:
                # Use stored Hessian from curve construction
                # NOTE: Stored Hessians already exclude prepended t=0 row (shape matches jac)
                hess_for_curve = foreign_model._hess
            else:
                # Fallback: Get from _cached_curve() (may have been computed above)
                if 'for_cache' not in locals():
                    for_cache = self._cached_curve(
                        tuple(foreign_model.swap_times),
                        foreign_model.swap_rates,
                        foreign_model.swap_times,
                        foreign_model.year_fracs,
                        foreign_model._interp_type
                    )
                hess_for_curve = for_cache["hess"][1:, :, :] if for_times[0] < 1e-6 else for_cache["hess"]

            # Check GAMMA precondition
            if hess_for_curve is None:
                raise LibError("GAMMA requested but foreign curve was not built with compute_gamma=True")

            # Foreign OIS GAMMA: Only direct effect on forward rates
            # IMPORTANT: XCCY curve is treated as FIXED when bumping foreign OIS rates.
            # Same rationale as for DELTA - we hold XCCY basis spreads fixed, so XCCY DFs
            # do not change. Therefore, only the direct effect on forward rates matters.

            # Compute GAMMA using SensitivityEngine (reuses pre-computed gradient)
            # Note: Result is in domestic currency, need FX adjustment
            gamma_foreign_obj = SensitivityEngine.compute_gamma(
                pv_fn=pv_for_original_dfs,
                dfs=for_ois_dfs_original,
                jac=jac_for_original,
                hess_curve=hess_for_curve,
                grad_dfs=grad_for_dfs_original,
                swap_times=foreign_model.swap_times,
                currency=derivative._domestic_currency,
                curve_type=derivative._foreign_floating_index
            )

            # Apply FX conversion: Foreign leg PV is in foreign currency, convert to domestic
            gammas_for = gamma_foreign_obj.risk_ladder / spot_fx

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
                    # NOTE: XCCY curve stores Hessian with prepended t=0 row (unlike OIS curves)
                    # Skip first row if curve has prepended t=0 to match original DFs
                    if xccy_times[0] < 1e-6:
                        if xccy_curve._hess_basis.ndim == 2:
                            hess_xccy_curve = xccy_curve._hess_basis[1:, :]  # Diagonal: (n_dfs, n_basis)
                        else:
                            hess_xccy_curve = xccy_curve._hess_basis[1:, :, :]  # Full: (n_dfs, n_basis, n_basis)
                    else:
                        hess_xccy_curve = xccy_curve._hess_basis

                    # Handle diagonal or full curve Hessian
                    if hess_xccy_curve.ndim == 2:
                        # Diagonal Hessian: shape (n_dfs, n_basis)
                        term2_diag = jnp.dot(grad_xccy_dfs_original, hess_xccy_curve)  # Shape: (n_basis,)
                        term2_xccy = jnp.diag(term2_diag)  # Shape: (n_basis, n_basis)
                    else:
                        # Full Hessian: shape (n_dfs, n_basis, n_basis)
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

        # FX01 calculation (FX sensitivity)
        fx_delta = None
        if RequestTypes.FX01 in reqs:
            # FX01 measures PV sensitivity to 1% move in spot FX
            # Since PV = PV_domestic + spot_fx * PV_foreign, the derivative is:
            # dPV/d(spot_fx) = PV_foreign
            # FX01 (for 1% move) = PV_foreign * spot_fx * 0.01
            #
            # Note: for_pv_scalar is already in foreign currency (GBP for USD/GBP swap)
            # Multiply by spot_fx to get domestic currency value, then by 0.01 for 1% sensitivity
            fx01_sensitivity = for_pv_scalar * spot_fx * 0.01

            fx_delta = FXDelta(
                sensitivity=fx01_sensitivity,
                spot_fx=spot_fx,
                currency=derivative._domestic_currency,
                domestic_currency=derivative._domestic_currency,
                foreign_currency=derivative._foreign_currency
            )

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

        return AnalyticsResult(value=value, risk=delta, gamma=gamma, cashflows=cashflows, fx_delta=fx_delta)

    def _compute_xccy_old(self, derivative, reqs):
        """Old array-based implementation - kept for DELTA/GAMMA future work."""
        # Get curves from model
        domestic_model = getattr(self.model.curves, derivative._domestic_floating_index.name)
        foreign_model = getattr(self.model.curves, derivative._foreign_floating_index.name)

        # Get XCCY curve and spot FX
        foreign_code = derivative._foreign_currency.name
        domestic_code = derivative._domestic_currency.name
        xccy_curve_name = f"{domestic_code}_{foreign_code}_BASIS"  # Match CurveTypes enum pattern

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
            if getattr(derivative._domestic_leg, '_notional_exchange', True):
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
            if getattr(derivative._foreign_leg, '_notional_exchange', True):
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
    

    def _preprocess_curve_points(self,
                                  swap_rates: list[float],
                                  year_fracs: list[list[float]],
                                  start_times: list[list[float]] = None) -> dict:
        """
        Pre-process curve points using Python operations (NOT JIT-compiled).

        This method handles deduplication and dependency graph construction using
        Python sets/dicts, which cannot be JIT-compiled. The actual numerical
        bootstrap is handled by _jit_bootstrap().

        This follows the same pattern as xccy_curve.py (lines 820-842) which uses
        Python preprocessing to avoid jnp.unique in differentiable computation.

        Args:
            swap_rates: Par swap rates for each maturity
            year_fracs: Year fractions for each swap's cashflows
            start_times: Start times for forward-starting instruments (optional).
                For spot-starting instruments, use 0.0. Defaults to None (all spot-starting).

        Returns:
            dict with JAX-ready arrays:
                - 'rates': Rates for each point
                - 'accs': Year fraction accumulations
                - 'prev_idxs': Index of previous point for each point
                - 'maturities': Exact maturity for each point
                - 'start_mats': Start maturities for forward-starting instruments (NEW)
        """
        # Default to all spot-starting if start_times not provided
        if start_times is None:
            start_times = [[0.0] * len(fracs) for fracs in year_fracs]
        # 1) Add t=0, df=1.0 point
        points = [{
            'maturity': 0.0,
            'maturity_key': 0.0,
            'acc': 0.0,
            'prev_mat': 0.0,
            'prev_key': None,
            'rate': swap_rates[0],
            'is_final': False,
            'swap_idx': -1,
            'start_mat': 0.0  # NEW: t=0 point is spot-starting
        }]

        # 2) Pre-expand ALL intermediate points with DEDUPLICATION
        # Uses Python set - fast for deduplication, but not JIT-compatible
        seen_keys = set([0.0])

        for i, (rate, fracs, starts) in enumerate(zip(swap_rates, year_fracs, start_times)):
            cumsum = 0.0  # Always start from 0 for consistency

            for j, (frac, start) in enumerate(zip(fracs, starts)):
                # For forward-starting instruments, reset cumsum to start time for first cashflow
                if j == 0 and start > 1e-6:
                    cumsum = start

                prev_cum = cumsum
                cumsum += frac
                key = round(cumsum, 4)  # 0.0001 years ≈ 0.9 hours - handles overnight deposits

                # DEDUPLICATION: "first occurrence wins"
                if key not in seen_keys:
                    # For forward-starting, first cashflow has no previous cashflow in the swap
                    # so prev_key should be None (bootstrap will use forward formula instead)
                    prev_key_val = None if j == 0 else round(prev_cum, 4)

                    points.append({
                        'maturity': cumsum,
                        'maturity_key': key,
                        'acc': frac,
                        'prev_mat': prev_cum,
                        'prev_key': prev_key_val,
                        'rate': rate,
                        'is_final': (j == len(fracs) - 1),
                        'swap_idx': i,
                        'start_mat': start  # When forward period begins (0.0 for spot-starting)
                    })
                    seen_keys.add(key)

        # 3) Sort by exact maturity
        sorted_points = sorted(points, key=lambda x: x['maturity'])

        # 4) Build maturity_key → first occurrence index mapping
        # Uses Python dict - fast, but not JIT-compatible
        maturity_lookup = {}
        for idx, p in enumerate(sorted_points):
            key = p['maturity_key']
            if key not in maturity_lookup:
                maturity_lookup[key] = idx

        # 5) Build prev_idx for each point
        for p in sorted_points:
            if p['prev_key'] is None:
                p['prev_idx'] = -1
            else:
                p['prev_idx'] = maturity_lookup.get(p['prev_key'], -1)

        # 6) Return as dict with arrays ready for JAX
        return {
            'rates': [p['rate'] for p in sorted_points],
            'accs': [p['acc'] for p in sorted_points],
            'prev_idxs': [p['prev_idx'] for p in sorted_points],
            'maturities': [p['maturity'] for p in sorted_points],
            'start_mats': [p['start_mat'] for p in sorted_points]  # NEW: For forward-starting instruments
        }

    @partial(jit, static_argnums=(0,))
    def _jit_bootstrap(self,
                      rates: jnp.ndarray,
                      accs: jnp.ndarray,
                      prev_idxs: jnp.ndarray,
                      start_mats: jnp.ndarray,
                      maturities: jnp.ndarray) -> jnp.ndarray:
        """
        JIT-compiled bootstrap computation (PURE JAX, no Python control flow).

        This is the performance-critical part that benefits from JIT compilation.
        All control flow and data structures have been pre-computed by
        _preprocess_curve_points().

        Supports BOTH spot-starting and forward-starting instruments:
        - Spot-starting (start_mat = 0): Uses standard bootstrap formula
        - Forward-starting (start_mat > 0): Interpolates DF at start_mat, then applies forward formula

        Args:
            rates: Swap rates for each point
            accs: Year fraction for each point
            prev_idxs: Index of previous point for dependency (-1 for t=0)
            start_mats: Start maturity for forward-starting instruments (0.0 for spot-starting)
            maturities: Exact maturity time for each point (for interpolation)

        Returns:
            all_dfs: Discount factors for all points
        """
        n_points = len(rates)

        def step(state, inputs):
            pv01_arr, dfs_arr = state
            i, rate, acc, prev_idx, start_mat, mat = inputs

            # Check if this is an interpolation-only point (dummy point for forward-starting)
            # These have rate=0 and acc=0 as markers
            is_interp_point = (jnp.abs(rate) < 1e-10) & (jnp.abs(acc) < 1e-10)

            # Mask valid points (0 to i-1) for interpolation
            valid_mask = jnp.arange(n_points) < i
            masked_mats = jnp.where(valid_mask, maturities, 1e10)
            masked_dfs = jnp.where(valid_mask, dfs_arr, 1.0)  # Use 1.0 for invalid to avoid log(0)

            # For interpolation-only points, just interpolate from existing DFs
            df_interp = jnp.interp(mat, masked_mats, masked_dfs, left=1.0, right=dfs_arr[i-1])

            # Compute DF for calibration points
            prev_pv01 = jnp.where(prev_idx < 0, 0.0, pv01_arr[prev_idx])

            # SPOT-STARTING formula (standard multi-cashflow bootstrap)
            df_spot = jnp.where(
                prev_idx < 0,
                1.0 / (1.0 + rate * acc),
                (1.0 - rate * prev_pv01) / (1.0 + rate * acc)
            )

            # FORWARD-STARTING formula: DF(end) = DF(start) / (1 + rate × acc)
            # Use iterative fixed-point solver to ensure bootstrap interpolation matches validation
            # This fixes the bootstrap-validation inconsistency that caused ~1e-6 errors

            # Initial guess using direct formula from incomplete curve
            zero_rates_guess = -jnp.log(masked_dfs) / jnp.maximum(masked_mats, 1e-15)
            last_valid_zero_rate = -jnp.log(dfs_arr[i-1]) / jnp.maximum(maturities[i-1], 1e-15)
            zero_rate_start_guess = jnp.interp(start_mat, masked_mats, zero_rates_guess, left=0.0, right=last_valid_zero_rate)
            df_start_guess = jnp.exp(-zero_rate_start_guess * start_mat)
            df_end_initial = df_start_guess / (1.0 + rate * acc)

            # Fixed-point iteration: df_end^{n+1} = df_start(df_end^n) / (1 + rate * acc)
            # where df_start(df_end) is interpolated from curve including df_end
            def fixed_point_step(df_end_guess):
                # Create temporary curve INCLUDING the candidate df_end at position i
                temp_dfs = dfs_arr.at[i].set(df_end_guess)

                # Mask to include points 0 to i
                temp_mask = jnp.arange(n_points) <= i
                temp_mats = jnp.where(temp_mask, maturities, 1e10)
                temp_dfs_masked = jnp.where(temp_mask, temp_dfs, 1.0)

                # Interpolate DF(start) from COMPLETE curve (includes df_end at position i)
                temp_zero_rates = -jnp.log(temp_dfs_masked) / jnp.maximum(temp_mats, 1e-15)
                zero_rate_start_temp = jnp.interp(start_mat, temp_mats, temp_zero_rates,
                                                   left=0.0, right=temp_zero_rates[i])
                df_start_temp = jnp.exp(-zero_rate_start_temp * start_mat)

                # Apply FRA formula: df_end = df_start / (1 + rate * acc)
                df_end_new = df_start_temp / (1.0 + rate * acc)

                return df_end_new

            # Run 10 fixed-point iterations (converges from ~1e-6 to <1e-10)
            df_forward = df_end_initial
            df_forward = fixed_point_step(df_forward)
            df_forward = fixed_point_step(df_forward)
            df_forward = fixed_point_step(df_forward)
            df_forward = fixed_point_step(df_forward)
            df_forward = fixed_point_step(df_forward)
            df_forward = fixed_point_step(df_forward)
            df_forward = fixed_point_step(df_forward)
            df_forward = fixed_point_step(df_forward)
            df_forward = fixed_point_step(df_forward)
            df_forward = fixed_point_step(df_forward)

            # Choose formula: interpolation, forward, or spot
            is_forward = start_mat > 1e-6
            df_calibrated = jnp.where(is_forward, df_forward, df_spot)
            df_i = jnp.where(is_interp_point, df_interp, df_calibrated)

            # Update PV01 (only for calibration points, not interpolation points)
            # For forward-starting instruments, interpolate PV01 at start_mat to get cumulative PV01
            pv01_start = jnp.interp(start_mat, masked_mats, pv01_arr, left=0.0, right=pv01_arr[i-1])
            pv01_forward = pv01_start + acc * df_i

            # Choose PV01 formula: spot (uses prev_pv01) or forward (uses pv01_start)
            pv01_calibrated = jnp.where(is_forward, pv01_forward, prev_pv01 + acc * df_i)
            pv01_i = jnp.where(is_interp_point, prev_pv01, pv01_calibrated)

            # Update state arrays
            new_pv01 = pv01_arr.at[i].set(pv01_i)
            new_dfs = dfs_arr.at[i].set(df_i)

            return (new_pv01, new_dfs), df_i

        # Initialize: t=0 has DF=1.0
        init_pv01 = jnp.zeros(n_points)
        init_dfs = jnp.zeros(n_points)
        init_dfs = init_dfs.at[0].set(1.0)

        # Run the scan (this gets JIT-compiled for 2-3x speedup)
        idxs = jnp.arange(n_points)
        scan_inputs = (idxs, rates, accs, prev_idxs, start_mats, maturities)
        (_, final_dfs), all_dfs = lax.scan(step, (init_pv01, init_dfs), scan_inputs)

        # Return DFs from scan outputs (each step returns df_i)
        return all_dfs

    def build_curve_ad(self,
                    swap_rates: list[float],
                    swap_times: list[float],
                    year_fracs: list[list[float]],
                    start_times: list[list[float]] = None
                    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Bootstraps an OIS curve via par-swap rates using JAX-compatible operations.

        PERFORMANCE: This method now uses JIT compilation for 2-3x speedup.
        - Python preprocessing (deduplication, sorting): ~1-5% of time
        - JAX JIT-compiled bootstrap (lax.scan): ~95-99% of time (ACCELERATED)

        Implementation pattern follows xccy_curve.py (lines 820-842) which uses
        Python preprocessing to avoid jnp.unique in differentiable computation.

        Args:
            swap_rates (list[float]): Par swap rates for each maturity
            swap_times (list[float]): Swap maturities in years
            year_fracs (list[list[float]]): Year fractions for each swap's cashflows
            start_times (list[list[float]], optional): Start times for forward-starting
                instruments (e.g., FRAs). For spot-starting instruments, use 0.0.
                Defaults to None (all instruments spot-starting).

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]:
                - all_maturities: All unique intermediate times including t=0
                - all_dfs: Discount factors at all intermediate times (df=1.0 at t=0)

        Implementation:
            1. Pre-process in Python (deduplication, dependency graph) - NOT JIT'd
            2. JIT-compiled numerical bootstrap (lax.scan) - FAST
            3. Return results as JAX arrays

        Note:
            Deduplication uses "first occurrence wins": when multiple swaps have cashflows
            at the same time (e.g., multiple 1Y+ swaps all have t=1.0), the FIRST swap to
            reach that time computes the DF using its rate. Subsequent swaps skip that time.
        """
        # Phase 1: Python preprocessing (fast, not the bottleneck)
        points_data = self._preprocess_curve_points(swap_rates, year_fracs, start_times)

        # Phase 2: JIT-compiled numerical bootstrap (THIS IS THE SPEEDUP)
        # Convert to JAX arrays
        rates = jnp.array(points_data['rates'])
        accs = jnp.array(points_data['accs'])
        prev_idxs = jnp.array(points_data['prev_idxs'], dtype=jnp.int32)
        start_mats = jnp.array(points_data['start_mats'])  # NEW: For forward-starting instruments
        all_maturities = jnp.array(points_data['maturities'])

        # Run JIT-compiled bootstrap (2-3x faster than non-JIT)
        all_dfs = self._jit_bootstrap(rates, accs, prev_idxs, start_mats, all_maturities)

        # Phase 3: Return results
        return all_maturities, all_dfs

    def _cached_curve(self, key, swap_rates, swap_times, year_fracs, interp_type, start_times=None, compute_gamma=True):
        """
        Bootstrap the curve once and cache DFS, Jacobian and optionally Hessian.

        Args:
            start_times: Start times for forward-starting instruments (optional)
            compute_gamma: If True, compute expensive Hessian for GAMMA calculations.
                          If False, skip Hessian (saves ~4-5s for large curves).
                          Default True for backward compatibility.
        """
        cache = self._curve_cache.get(key)
        if cache is not None:
            return cache

        # Build curve to get both times and DFs (including all intermediate points)
        rates = jnp.array(swap_rates)
        times, dfs = self.build_curve_ad(rates, swap_times, year_fracs, start_times)

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
            _, dfs_out = self.build_curve_ad(r, swap_times, year_fracs, start_times)
            return dfs_out

        jac_original = jacrev(build_dfs_original)(rates)

        # Hessian computation is expensive (~4-5s for large curves)
        # Only compute if GAMMA will be requested
        if compute_gamma:
            hess_original = hessian(build_dfs_original)(rates)
        else:
            hess_original = None

        # If we prepended time=0, add a row of zeros to Jacobian and Hessian
        # because DF(t=0) = 1.0 is constant (zero gradient w.r.t. all rates)
        if prepended_t0:
            n_rates = len(rates)
            zero_row = jnp.zeros((1, n_rates))
            jac = jnp.concatenate([zero_row, jac_original], axis=0)

            # For Hessian: prepend zeros for the time=0 point (only if computed)
            if hess_original is not None:
                zero_matrix = jnp.zeros((1, n_rates, n_rates))
                hess = jnp.concatenate([zero_matrix, hess_original], axis=0)
            else:
                hess = None
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

    def _fixed_leg_jax(self,
                       dfs,                          # Discount factors [N]
                       times,                        # Times [N]
                       disc_interp_type,             # Interpolation type
                       payment_times,                # [M] - payment dates
                       payments,                     # [M] - fixed payment amounts
                       principal: float,             # scalar - principal payment
                       leg_sign: float,              # +1 or -1
                       value_time: float,            # scalar - valuation time
                       notional_exchange=False,      # bool - enable notional exchanges
                       notional_exchange_amount=0.0, # scalar - notional amount
                       effective_time=0.0,           # scalar - effective date time
                       maturity_time=0.0             # scalar - maturity date time
                       ):
        """
        JAX-compatible fixed leg pricing with optional notional exchanges for XCCY swaps.

        Similar to _price_fixed_leg_jax() but with XCCY notional exchange support.
        Used for AD-based GAMMA computation in _compute_xccy().

        Args:
            dfs: Discount factors array
            times: Time grid for interpolation
            disc_interp_type: Interpolation method
            payment_times: Payment date times [M]
            payments: Fixed payment amounts [M]
            principal: Principal payment (typically 0 for swaps)
            leg_sign: +1 for receive, -1 for pay
            value_time: Valuation date time
            notional_exchange: Enable XCCY notional exchanges
            notional_exchange_amount: Notional amount for exchanges
            effective_time: Effective date time (for initial exchange)
            maturity_time: Maturity date time (for final exchange)

        Returns:
            Scalar PV of fixed leg with notional exchanges

        Mathematical formula:
            PV = sign × [Σ(payment_i × DF(t_i)) + principal × DF(t_final)
                        - notional × DF(t_eff) + notional × DF(t_mat)]
        """
        from cavour.market.curves.interpolator_ad import InterpolatorAd

        interp = InterpolatorAd(disc_interp_type)

        # Discount factor at valuation date
        df_val = jnp.atleast_1d(interp.simple_interpolate(value_time, times, dfs, disc_interp_type.value))

        # Discount factors at payment dates
        df_pmts = jnp.atleast_1d(interp.simple_interpolate(payment_times, times, dfs, disc_interp_type.value))

        # Mask for future payments
        mask = payment_times > value_time  # [M]
        mask = jnp.broadcast_to(mask, df_pmts.shape)

        # Relative discount factors
        df_rel = df_pmts / df_val[..., None]  # [..., M]

        # PV of fixed coupons
        pv_coupons = jnp.where(mask, payments * df_rel, 0.0)  # [..., M]

        # PV of principal (typically 0 for swaps)
        final_mask = mask[..., -1]
        final_df_rel = df_rel[..., -1]
        pv_prin = jnp.where(final_mask, principal * final_df_rel, 0.0)

        # Notional exchanges (for XCCY swaps)
        # Start exchange: -notional at effective_dt (outflow)
        df_effective = jnp.atleast_1d(interp.simple_interpolate(effective_time, times, dfs, disc_interp_type.value))
        df_effective_rel = df_effective / df_val
        pv_start_exchange = jnp.where(effective_time >= value_time,
                                      -notional_exchange_amount * df_effective_rel,
                                      0.0)

        # End exchange: +notional at maturity_dt (inflow)
        df_maturity = jnp.atleast_1d(interp.simple_interpolate(maturity_time, times, dfs, disc_interp_type.value))
        df_maturity_rel = df_maturity / df_val
        pv_end_exchange = jnp.where(maturity_time >= value_time,
                                    notional_exchange_amount * df_maturity_rel,
                                    0.0)

        # Multiply by boolean flag (True→1.0, False→0.0 in JAX)
        # Convert boolean to float explicitly
        notional_exchange_float = jnp.where(notional_exchange, 1.0, 0.0)
        pv_notional_exchange = jnp.squeeze(pv_start_exchange + pv_end_exchange) * notional_exchange_float

        # Total PV
        pv_coupons_sum = jnp.sum(pv_coupons, axis=-1)
        leg_pv = pv_coupons_sum + pv_prin + pv_notional_exchange

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

        # Use SensitivityEngine for DELTA and GAMMA computation (centralized implementation)
        from cavour.market.sensitivity import SensitivityEngine

        need_both = RequestTypes.DELTA in requests and RequestTypes.GAMMA in requests
        if need_both:
            # Compute both efficiently (shares gradient computation)
            delta, gamma = SensitivityEngine.compute_delta_gamma(
                pv_fn=pv_fn,
                dfs=dfs,
                jac=jac,
                hess_curve=hess_curve,
                swap_times=swap_times,
                currency=fixed_leg_details._currency,
                curve_type=fixed_leg_details._floating_index
            )
            out["delta"] = delta
            out["gamma"] = gamma
        else:
            # Compute only what's requested
            if RequestTypes.DELTA in requests:
                out["delta"] = SensitivityEngine.compute_delta(
                    pv_fn=pv_fn,
                    dfs=dfs,
                    jac=jac,
                    swap_times=swap_times,
                    currency=fixed_leg_details._currency,
                    curve_type=fixed_leg_details._floating_index
                )

            if RequestTypes.GAMMA in requests:
                out["gamma"] = SensitivityEngine.compute_gamma(
                    pv_fn=pv_fn,
                    dfs=dfs,
                    jac=jac,
                    hess_curve=hess_curve,
                    grad_dfs=None,  # Will be computed inside
                    swap_times=swap_times,
                    currency=fixed_leg_details._currency,
                    curve_type=fixed_leg_details._floating_index
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
        # IMPORTANT: Always compute, then multiply by notional_exchange boolean flag
        # (cannot use Python if with JAX tracers during vmap)

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

        # Multiply by boolean flag (True→1.0, False→0.0 in JAX)
        pv_notional_exchange = jnp.squeeze(pv_start_exchange + pv_end_exchange) * notional_exchange

        # j) aggregate and apply sign
        pv_coupons_sum = jnp.sum(pv_coupons, axis=-1)
        leg_pv     = pv_coupons_sum + pv_prin + pv_notional_exchange  # [...]

        return leg_sign * leg_pv


    def _xccy_pv_pure(self,
                     dom_dfs, dom_times, dom_interp_type,
                     for_dfs, for_times, for_interp_type,
                     xccy_dfs, xccy_times, xccy_interp_type,
                     dom_payment_times, dom_start_times, dom_end_times,
                     dom_alphas, dom_spreads, dom_notionals,
                     dom_principal, dom_leg_sign,
                     dom_notional_exchange, dom_effective_time, dom_maturity_time,
                     for_payment_times, for_start_times, for_end_times,
                     for_alphas, for_spreads, for_notionals,
                     for_principal, for_leg_sign,
                     for_notional_exchange, for_effective_time, for_maturity_time,
                     value_time, spot_fx):
        """
        Pure function to compute XCCY swap PV (JAX-compatible, no side effects).

        This is the core pricing function designed for vectorization via vmap.
        Takes all inputs as explicit parameters with no access to self.model or any mutable state.

        Architecture:
            - Domestic leg: Single-curve pricing (domestic OIS for both discount and forward)
            - Foreign leg: Dual-curve pricing (XCCY curve for discount, foreign OIS for forward)
            - FX conversion: Foreign PV converted to domestic currency via spot_fx

        Args:
            Curve arrays (shared across batch):
                dom_dfs, dom_times: Domestic OIS discount factors and times
                dom_interp_type: Interpolation type for domestic curve
                for_dfs, for_times: Foreign OIS discount factors and times (for forward rates)
                for_interp_type: Interpolation type for foreign curve
                xccy_dfs, xccy_times: XCCY curve discount factors and times (for foreign discounting)
                xccy_interp_type: Interpolation type for XCCY curve

            Domestic leg parameters (per-swap, will be batched):
                dom_payment_times: Payment date times [M_dom]
                dom_start_times: Accrual start times [M_dom]
                dom_end_times: Accrual end times [M_dom]
                dom_alphas: Year fractions [M_dom]
                dom_spreads: Spread on each cashflow [M_dom]
                dom_notionals: Notional on each cashflow [M_dom]
                dom_principal: Principal amount (scalar)
                dom_leg_sign: +1 (receive) or -1 (pay)
                dom_notional_exchange: Boolean flag for notional exchanges
                dom_effective_time: Time to effective date (for start exchange)
                dom_maturity_time: Time to maturity date (for end exchange)

            Foreign leg parameters (per-swap, will be batched):
                for_payment_times: Payment date times [M_for]
                for_start_times: Accrual start times [M_for]
                for_end_times: Accrual end times [M_for]
                for_alphas: Year fractions [M_for]
                for_spreads: Spread on each cashflow [M_for]
                for_notionals: Notional on each cashflow [M_for]
                for_principal: Principal amount (scalar)
                for_leg_sign: +1 (receive) or -1 (pay)
                for_notional_exchange: Boolean flag for notional exchanges
                for_effective_time: Time to effective date (for start exchange)
                for_maturity_time: Time to maturity date (for end exchange)

            Scalars (shared across batch):
                value_time: Valuation time
                spot_fx: FX spot rate (domestic/foreign, e.g., USD/GBP)

        Returns:
            PV in domestic currency (scalar)

        Performance:
            - Pure JAX function: JIT compilable
            - Vectorizable via vmap for batch pricing
            - No Python side effects: no dictionary lookups, no conditionals on RequestTypes

        Note:
            Variable-length cashflows handled via padding in caller:
            - Pad dom/for arrays to max_cashflows with zeros
            - Validity handled automatically by payment_times >= value_time mask in _float_leg_jax
        """
        import jax.numpy as jnp

        # Domestic leg PV (single-curve: domestic OIS for both discount and forward)
        dom_pv = self._float_leg_jax(
            dfs=dom_dfs,
            times=dom_times,
            disc_interp_type=dom_interp_type,
            idx_interp_type=dom_interp_type,
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
            notional_exchange=dom_notional_exchange,
            notional_exchange_amount=dom_notionals[0],  # Use first notional for exchanges
            effective_time=dom_effective_time,
            maturity_time=dom_maturity_time
        )

        # Foreign leg PV (dual-curve: XCCY for discount, foreign OIS for forward rates)
        for_pv = self._float_leg_jax(
            dfs=xccy_dfs,  # XCCY curve for discounting
            times=xccy_times,
            disc_interp_type=xccy_interp_type,
            idx_interp_type=for_interp_type,
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
            notional_exchange=for_notional_exchange,
            notional_exchange_amount=for_notionals[0],  # Use first notional for exchanges
            effective_time=for_effective_time,
            maturity_time=for_maturity_time
        )

        # Total PV: domestic PV + foreign PV converted to domestic currency
        # spot_fx is domestic/foreign (e.g., USD/GBP: 1.27 means 1 GBP = 1.27 USD)
        # for_pv is in foreign currency (GBP), multiply by spot_fx to convert to domestic (USD)
        total_pv = dom_pv + for_pv * spot_fx

        # Return scalar (squeeze any extra dimensions from JAX arrays)
        return jnp.squeeze(total_pv)


    def _xccy_delta_pure(self,
                        dom_dfs, dom_times, dom_interp_type,
                        for_dfs, for_times, for_interp_type,
                        xccy_dfs, xccy_times, xccy_interp_type,
                        dom_jac, for_jac, xccy_jac_basis,
                        dom_payment_times, dom_start_times, dom_end_times,
                        dom_alphas, dom_spreads, dom_notionals,
                        dom_principal, dom_leg_sign,
                        dom_notional_exchange, dom_effective_time, dom_maturity_time,
                        for_payment_times, for_start_times, for_end_times,
                        for_alphas, for_spreads, for_notionals,
                        for_principal, for_leg_sign,
                        for_notional_exchange, for_effective_time, for_maturity_time,
                        value_time, spot_fx):
        """
        Compute XCCY swap DELTA for a single swap (JAX-compatible, vmap-ready).

        This pure function computes first-order sensitivities (DELTA) to:
        - Domestic OIS curve rates
        - Foreign OIS curve rates
        - XCCY basis spreads

        Uses automatic differentiation to compute gradients w.r.t. discount factors,
        then applies chain rule with pre-computed Jacobians.

        Architecture:
        1. Define PV functions for each curve (domestic, foreign, xccy)
        2. Compute gradients: grad(PV, DFs) for each curve
        3. Chain rule: delta = grad(PV, DFs) · Jacobian(DFs, rates)
        4. Convert to bp units and domestic currency

        Args:
            Curve arrays (shared across batch):
                dom_dfs, dom_times: Domestic OIS discount factors and times
                dom_interp_type: Interpolation type for domestic curve
                for_dfs, for_times: Foreign OIS discount factors and times
                for_interp_type: Interpolation type for foreign curve
                xccy_dfs, xccy_times: XCCY curve discount factors and times
                xccy_interp_type: Interpolation type for XCCY curve

            Jacobians (shared across batch):
                dom_jac: [n_dfs, n_rates] Jacobian d(DFs)/d(rates) for domestic OIS
                for_jac: [n_dfs, n_rates] Jacobian d(DFs)/d(rates) for foreign OIS
                xccy_jac_basis: [n_dfs, n_basis] Jacobian d(DFs)/d(basis_spreads) for XCCY

            Swap parameters (per-swap, batched via vmap):
                dom_* : Domestic leg parameters (payment times, alphas, spreads, etc.)
                for_* : Foreign leg parameters
                value_time: Valuation time
                spot_fx: FX spot rate (domestic/foreign)

        Returns:
            Tuple of (delta_dom, delta_for, delta_basis):
                delta_dom: [n_dom_rates] - domestic OIS delta in USD/bp
                delta_for: [n_for_rates] - foreign OIS delta in USD/bp
                delta_basis: [n_basis] - XCCY basis delta in USD/bp

        Performance:
            - Pure JAX function: JIT compilable, vmap-compatible
            - Shared Jacobians amortize overhead across batch
            - Per-swap gradients computed in parallel via vmap
        """
        from jax import grad
        import jax.numpy as jnp

        # Define PV functions for each curve (same pattern as sequential _compute_xccy)

        # Domestic leg PV as function of domestic DFs
        def pv_dom_fn(dom_dfs_var):
            return self._float_leg_jax(
                dfs=dom_dfs_var, times=dom_times,
                disc_interp_type=dom_interp_type,
                idx_interp_type=dom_interp_type,
                payment_times=dom_payment_times,
                start_times=dom_start_times, end_times=dom_end_times,
                pay_alphas=dom_alphas, spreads=dom_spreads,
                notionals=dom_notionals, principal=dom_principal,
                leg_sign=dom_leg_sign, value_time=value_time,
                first_fixing_rate=0.0, override_first=False,
                idx_times=None, idx_dfs=None,
                notional_exchange=dom_notional_exchange,
                notional_exchange_amount=dom_notionals[0],
                effective_time=dom_effective_time,
                maturity_time=dom_maturity_time
            )

        # Foreign leg PV as function of foreign OIS DFs (for forward rate sensitivity)
        def pv_for_fn(for_ois_dfs_var):
            return self._float_leg_jax(
                dfs=xccy_dfs, times=xccy_times,  # XCCY curve for discounting (FIXED)
                disc_interp_type=xccy_interp_type,
                idx_interp_type=for_interp_type,
                payment_times=for_payment_times,
                start_times=for_start_times, end_times=for_end_times,
                pay_alphas=for_alphas, spreads=for_spreads,
                notionals=for_notionals, principal=for_principal,
                leg_sign=for_leg_sign, value_time=value_time,
                first_fixing_rate=0.0, override_first=False,
                idx_times=for_times, idx_dfs=for_ois_dfs_var,  # Foreign OIS DFs (VARIABLE)
                notional_exchange=for_notional_exchange,
                notional_exchange_amount=for_notionals[0],
                effective_time=for_effective_time,
                maturity_time=for_maturity_time
            )

        # Foreign leg PV as function of XCCY DFs (for basis spread sensitivity)
        def pv_xccy_fn(xccy_dfs_var):
            return self._float_leg_jax(
                dfs=xccy_dfs_var, times=xccy_times,  # XCCY curve for discounting (VARIABLE)
                disc_interp_type=xccy_interp_type,
                idx_interp_type=for_interp_type,
                payment_times=for_payment_times,
                start_times=for_start_times, end_times=for_end_times,
                pay_alphas=for_alphas, spreads=for_spreads,
                notionals=for_notionals, principal=for_principal,
                leg_sign=for_leg_sign, value_time=value_time,
                first_fixing_rate=0.0, override_first=False,
                idx_times=for_times, idx_dfs=for_dfs,  # Foreign OIS DFs (FIXED)
                notional_exchange=for_notional_exchange,
                notional_exchange_amount=for_notionals[0],
                effective_time=for_effective_time,
                maturity_time=for_maturity_time
            )

        # Use wrapper functions to handle prepended t=0 (same pattern as sequential _compute_xccy)
        # IMPORTANT: DF(t≈0) = 1.0 is a boundary condition, NOT a curve parameter.
        # Jacobians are w.r.t. "original" DFs (excluding prepended point).

        # Extract original DFs (excluding prepended t=0 if present)
        dom_dfs_original = dom_dfs[1:] if dom_times[0] < 1e-6 else dom_dfs
        for_dfs_original = for_dfs[1:] if for_times[0] < 1e-6 else for_dfs
        xccy_dfs_original = xccy_dfs[1:] if xccy_times[0] < 1e-6 else xccy_dfs

        # Wrapper functions that prepend t=0 before calling PV function
        def pv_dom_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs]) if dom_times[0] < 1e-6 else original_dfs
            return pv_dom_fn(full_dfs)

        def pv_for_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs]) if for_times[0] < 1e-6 else original_dfs
            return pv_for_fn(full_dfs)

        def pv_xccy_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs]) if xccy_times[0] < 1e-6 else original_dfs
            return pv_xccy_fn(full_dfs)

        # Compute gradients w.r.t. ORIGINAL DFs only (excluding DF(0)=1.0)
        grad_dom_original = grad(lambda d: jnp.squeeze(pv_dom_original_dfs(d)))(dom_dfs_original)
        grad_for_original = grad(lambda d: jnp.squeeze(pv_for_original_dfs(d)))(for_dfs_original)
        grad_xccy_original = grad(lambda d: jnp.squeeze(pv_xccy_original_dfs(d)))(xccy_dfs_original)

        # Apply chain rule with Jacobians
        delta_dom_rates_raw = jnp.dot(grad_dom_original, dom_jac)
        delta_for_rates_raw = jnp.dot(grad_for_original, for_jac)
        delta_basis_rates_raw = jnp.dot(grad_xccy_original, xccy_jac_basis)

        # Convert to bp units (rates stored in decimal, 1bp = 0.0001) and domestic currency
        # Domestic leg: already in domestic currency (USD)
        delta_dom = delta_dom_rates_raw * 1e-4

        # Foreign leg: delta is d(PV_foreign)/d(rate), need to convert to domestic currency
        # Foreign PV is in foreign currency (GBP), multiply by spot_fx (USD/GBP) to get USD
        delta_for = delta_for_rates_raw * 1e-4 * spot_fx

        # XCCY basis: delta is d(PV_foreign)/d(basis), need to convert to domestic currency
        delta_basis = delta_basis_rates_raw * 1e-4 * spot_fx

        return delta_dom, delta_for, delta_basis

    def _xccy_gamma_pure(self,
                        dom_dfs, dom_times, dom_interp_type,
                        for_dfs, for_times, for_interp_type,
                        xccy_dfs, xccy_times, xccy_interp_type,
                        dom_jac, for_jac, xccy_jac_basis,
                        dom_hess, for_hess, xccy_hess_basis,
                        dom_payment_times, dom_start_times, dom_end_times,
                        dom_alphas, dom_spreads, dom_notionals,
                        dom_principal, dom_leg_sign,
                        dom_notional_exchange, dom_effective_time, dom_maturity_time,
                        for_payment_times, for_start_times, for_end_times,
                        for_alphas, for_spreads, for_notionals,
                        for_principal, for_leg_sign,
                        for_notional_exchange, for_effective_time, for_maturity_time,
                        value_time, spot_fx):
        """
        Compute XCCY swap GAMMA for a single swap (JAX-compatible, vmap-ready).

        This pure function computes second-order sensitivities (GAMMA) to:
        - Domestic OIS curve rates
        - Foreign OIS curve rates
        - XCCY basis spreads

        Uses automatic differentiation to compute Hessians w.r.t. discount factors,
        then applies chain rule with pre-computed Jacobians and curve Hessians.

        Architecture:
        1. Define PV functions for each curve (domestic, foreign, xccy)
        2. Compute gradients: grad(PV, DFs) for each curve
        3. Compute Hessians: hess(PV, DFs) for each curve
        4. Chain rule: gamma = jac^T @ hess(PV, DFs) @ jac + sum(grad * hess_curve)
        5. Convert to bp² units and domestic currency

        Args:
            Curve arrays (shared across batch):
                dom_dfs, dom_times: Domestic OIS discount factors and times
                dom_interp_type: Interpolation type for domestic curve
                for_dfs, for_times: Foreign OIS discount factors and times
                for_interp_type: Interpolation type for foreign curve
                xccy_dfs, xccy_times: XCCY curve discount factors and times
                xccy_interp_type: Interpolation type for XCCY curve

            Jacobians (shared across batch):
                dom_jac: [n_dfs, n_rates] Jacobian d(DFs)/d(rates) for domestic OIS
                for_jac: [n_dfs, n_rates] Jacobian d(DFs)/d(rates) for foreign OIS
                xccy_jac_basis: [n_dfs, n_basis] Jacobian d(DFs)/d(basis_spreads) for XCCY

            Hessians (shared across batch):
                dom_hess: [n_dfs, n_rates, n_rates] Hessian d²(DFs)/d(rates)² for domestic OIS
                for_hess: [n_dfs, n_rates, n_rates] Hessian d²(DFs)/d(rates)² for foreign OIS
                xccy_hess_basis: [n_dfs, n_basis, n_basis] Hessian d²(DFs)/d(basis_spreads)²

            Swap parameters (per-swap, batched via vmap):
                dom_* : Domestic leg parameters (payment times, alphas, spreads, etc.)
                for_* : Foreign leg parameters
                value_time: Valuation time
                spot_fx: FX spot rate (domestic/foreign)

        Returns:
            Tuple of (gamma_dom, gamma_for, gamma_basis):
                gamma_dom: [n_dom_rates, n_dom_rates] - domestic OIS gamma in USD/bp²
                gamma_for: [n_for_rates, n_for_rates] - foreign OIS gamma in USD/bp²
                gamma_basis: [n_basis, n_basis] - XCCY basis gamma in USD/bp²

        Performance:
            - Pure JAX function: JIT compilable, vmap-compatible
            - Shared Jacobians/Hessians amortize overhead across batch
            - Per-swap Hessians computed in parallel via vmap
        """
        from jax import grad, hessian
        import jax.numpy as jnp

        # Define PV functions for each curve (same pattern as _xccy_delta_pure)

        # Domestic leg PV as function of domestic DFs
        def pv_dom_fn(dom_dfs_var):
            return self._float_leg_jax(
                dfs=dom_dfs_var, times=dom_times,
                disc_interp_type=dom_interp_type,
                idx_interp_type=dom_interp_type,
                payment_times=dom_payment_times,
                start_times=dom_start_times, end_times=dom_end_times,
                pay_alphas=dom_alphas, spreads=dom_spreads,
                notionals=dom_notionals, principal=dom_principal,
                leg_sign=dom_leg_sign, value_time=value_time,
                first_fixing_rate=0.0, override_first=False,
                idx_times=None, idx_dfs=None,
                notional_exchange=dom_notional_exchange,
                notional_exchange_amount=dom_notionals[0],
                effective_time=dom_effective_time,
                maturity_time=dom_maturity_time
            )

        # Foreign leg PV as function of foreign OIS DFs (for forward rate sensitivity)
        def pv_for_fn(for_ois_dfs_var):
            return self._float_leg_jax(
                dfs=xccy_dfs, times=xccy_times,  # XCCY curve for discounting (FIXED)
                disc_interp_type=xccy_interp_type,
                idx_interp_type=for_interp_type,
                payment_times=for_payment_times,
                start_times=for_start_times, end_times=for_end_times,
                pay_alphas=for_alphas, spreads=for_spreads,
                notionals=for_notionals, principal=for_principal,
                leg_sign=for_leg_sign, value_time=value_time,
                first_fixing_rate=0.0, override_first=False,
                idx_times=for_times, idx_dfs=for_ois_dfs_var,  # Foreign OIS DFs (VARIABLE)
                notional_exchange=for_notional_exchange,
                notional_exchange_amount=for_notionals[0],
                effective_time=for_effective_time,
                maturity_time=for_maturity_time
            )

        # Foreign leg PV as function of XCCY DFs (for basis spread sensitivity)
        def pv_xccy_fn(xccy_dfs_var):
            return self._float_leg_jax(
                dfs=xccy_dfs_var, times=xccy_times,  # XCCY curve for discounting (VARIABLE)
                disc_interp_type=xccy_interp_type,
                idx_interp_type=for_interp_type,
                payment_times=for_payment_times,
                start_times=for_start_times, end_times=for_end_times,
                pay_alphas=for_alphas, spreads=for_spreads,
                notionals=for_notionals, principal=for_principal,
                leg_sign=for_leg_sign, value_time=value_time,
                first_fixing_rate=0.0, override_first=False,
                idx_times=for_times, idx_dfs=for_dfs,  # Foreign OIS DFs (FIXED)
                notional_exchange=for_notional_exchange,
                notional_exchange_amount=for_notionals[0],
                effective_time=for_effective_time,
                maturity_time=for_maturity_time
            )

        # Use wrapper functions to handle prepended t=0 (same pattern as _xccy_delta_pure)
        # IMPORTANT: DF(t≈0) = 1.0 is a boundary condition, NOT a curve parameter.
        # Jacobians/Hessians are w.r.t. "original" DFs (excluding prepended point).

        # Extract original DFs (excluding prepended t=0 if present)
        dom_dfs_original = dom_dfs[1:] if dom_times[0] < 1e-6 else dom_dfs
        for_dfs_original = for_dfs[1:] if for_times[0] < 1e-6 else for_dfs
        xccy_dfs_original = xccy_dfs[1:] if xccy_times[0] < 1e-6 else xccy_dfs

        # Wrapper functions that prepend t=0 before calling PV function
        def pv_dom_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs]) if dom_times[0] < 1e-6 else original_dfs
            return pv_dom_fn(full_dfs)

        def pv_for_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs]) if for_times[0] < 1e-6 else original_dfs
            return pv_for_fn(full_dfs)

        def pv_xccy_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs]) if xccy_times[0] < 1e-6 else original_dfs
            return pv_xccy_fn(full_dfs)

        # Compute gradients and Hessians w.r.t. ORIGINAL DFs only (excluding DF(0)=1.0)

        # Domestic OIS GAMMA
        grad_dom_original = grad(lambda d: jnp.squeeze(pv_dom_original_dfs(d)))(dom_dfs_original)
        hess_dom_dfs = hessian(lambda d: jnp.squeeze(pv_dom_original_dfs(d)))(dom_dfs_original)

        # Chain rule: gamma = jac^T @ hess_pv_dfs @ jac + sum(grad * hess_curve)
        term1_dom = dom_jac.T @ hess_dom_dfs @ dom_jac

        # Handle diagonal or full curve Hessian
        if dom_hess.ndim == 2:
            # Diagonal Hessian: shape (n_dfs, n_rates)
            term2_diag = jnp.dot(grad_dom_original, dom_hess)  # Shape: (n_rates,)
            term2_dom = jnp.diag(term2_diag)  # Shape: (n_rates, n_rates)
        else:
            # Full Hessian: shape (n_dfs, n_rates, n_rates)
            term2_dom = jnp.sum(grad_dom_original[:, None, None] * dom_hess, axis=0)

        gamma_dom_matrix = term1_dom + term2_dom

        # Convert to USD per bp² (1bp = 0.0001 → bp² = 1e-8)
        gamma_dom = gamma_dom_matrix * 1e-8

        # Foreign OIS GAMMA (direct effect through forward rates)
        grad_for_original = grad(lambda d: jnp.squeeze(pv_for_original_dfs(d)))(for_dfs_original)
        hess_for_dfs = hessian(lambda d: jnp.squeeze(pv_for_original_dfs(d)))(for_dfs_original)

        # Chain rule (direct effect only - XCCY curve held fixed)
        term1_for = for_jac.T @ hess_for_dfs @ for_jac

        # Handle diagonal or full curve Hessian
        if for_hess.ndim == 2:
            # Diagonal Hessian: shape (n_dfs, n_rates)
            term2_diag = jnp.dot(grad_for_original, for_hess)  # Shape: (n_rates,)
            term2_for = jnp.diag(term2_diag)  # Shape: (n_rates, n_rates)
        else:
            # Full Hessian: shape (n_dfs, n_rates, n_rates)
            term2_for = jnp.sum(grad_for_original[:, None, None] * for_hess, axis=0)

        gamma_for_matrix = term1_for + term2_for

        # Convert to USD per bp²
        # NOTE: Sequential code divides by spot_fx (matching line 2515 in _compute_xccy)
        gamma_for = gamma_for_matrix * 1e-8 / spot_fx

        # XCCY Basis GAMMA
        grad_xccy_original = grad(lambda d: jnp.squeeze(pv_xccy_original_dfs(d)))(xccy_dfs_original)
        hess_xccy_dfs = hessian(lambda d: jnp.squeeze(pv_xccy_original_dfs(d)))(xccy_dfs_original)

        # Chain rule (with curve Hessian if available)
        term1_xccy = xccy_jac_basis.T @ hess_xccy_dfs @ xccy_jac_basis

        # Handle diagonal or full curve Hessian
        if xccy_hess_basis.ndim == 2:
            # Diagonal Hessian: shape (n_dfs, n_basis)
            term2_diag = jnp.dot(grad_xccy_original, xccy_hess_basis)  # Shape: (n_basis,)
            term2_xccy = jnp.diag(term2_diag)  # Shape: (n_basis, n_basis)
        else:
            # Full Hessian: shape (n_dfs, n_basis, n_basis)
            term2_xccy = jnp.sum(grad_xccy_original[:, None, None] * xccy_hess_basis, axis=0)

        gamma_basis_matrix = term1_xccy + term2_xccy

        # Convert to USD per bp²
        # NOTE: Sequential code divides by spot_fx (matching line 2601 in _compute_xccy)
        gamma_basis = gamma_basis_matrix * 1e-8 / spot_fx

        return gamma_dom, gamma_for, gamma_basis

    def _ois_pv_pure(self,
                    dfs, times, interp_type,
                    fixed_payment_times, fixed_payments, fixed_principal, fixed_leg_sign,
                    float_payment_times, float_start_times, float_end_times,
                    float_alphas, float_spreads, float_notionals, float_principal, float_leg_sign,
                    value_time):
        """
        Compute OIS swap PV for a single swap (JAX-compatible, vmap-ready).

        Pure function for batched OIS pricing with natural currency collateral (single-curve).

        Args:
            dfs: Discount factors [N]
            times: Curve times [N]
            interp_type: Interpolation method

            fixed_payment_times: Fixed leg payment times [M_fixed]
            fixed_payments: Fixed leg payment amounts [M_fixed]
            fixed_principal: Final principal for fixed leg (scalar)
            fixed_leg_sign: +1 (receive fixed) or -1 (pay fixed)

            float_payment_times: Float leg payment times [M_float]
            float_start_times: Float leg period start times [M_float]
            float_end_times: Float leg period end times [M_float]
            float_alphas: Float leg year fractions [M_float]
            float_spreads: Float leg spreads [M_float]
            float_notionals: Float leg notionals [M_float]
            float_principal: Final principal for float leg (scalar)
            float_leg_sign: +1 (receive float) or -1 (pay float)

            value_time: Valuation time (scalar)

        Returns:
            float: Total swap PV (fixed_pv + float_pv)

        Notes:
            - Natural currency pricing uses single curve for both discounting and projection
            - Fixed leg: Discounts known fixed coupon payments
            - Float leg: Projects forward rates and discounts, all from same OIS curve
            - No notional exchange (standard OIS structure)
            - No FX conversion (single currency)

        Performance:
            - Pure JAX function: JIT compilable, vmap-compatible
            - Simpler than XCCY: one fixed leg + one float leg vs two float legs
            - No notional exchange overhead
            - Shared curves amortize overhead across batch
        """
        import jax.numpy as jnp

        # Price fixed leg
        # Reuses existing _price_fixed_leg_jax pure function
        fixed_pv = self._price_fixed_leg_jax(
            dfs=dfs,
            times=times,
            interp_type=interp_type,
            payment_times=fixed_payment_times,
            payments=fixed_payments,
            principal=fixed_principal,
            notional=1.0,  # Notional already baked into payments
            leg_sign=fixed_leg_sign,
            value_time=value_time
        )

        # Price floating leg (single-curve mode)
        # Reuses existing _float_leg_jax pure function
        # For natural currency OIS: same curve used for discounting and forward rate projection
        float_pv = self._float_leg_jax(
            dfs=dfs,
            times=times,
            disc_interp_type=interp_type,
            idx_interp_type=interp_type,
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
            idx_times=None,  # Single-curve: use same DFs for forward rates
            idx_dfs=None,
            notional_exchange=False,  # No notional exchange for OIS
            notional_exchange_amount=0.0,
            effective_time=0.0,
            maturity_time=0.0
        )

        # Combine legs
        return fixed_pv + float_pv

    def _ois_delta_pure(self,
                       dfs, times, interp_type, jac,
                       fixed_payment_times, fixed_payments, fixed_principal, fixed_leg_sign,
                       float_payment_times, float_start_times, float_end_times,
                       float_alphas, float_spreads, float_notionals, float_principal, float_leg_sign,
                       value_time):
        """
        Compute OIS swap DELTA for a single swap (JAX-compatible, vmap-ready).

        Pure function for batched OIS DELTA computation using automatic differentiation.

        Args:
            dfs: Discount factors [N]
            times: Curve times [N]
            interp_type: Interpolation method
            jac: Jacobian d(DFs)/d(rates) [N, M] - pre-computed from curve construction

            [Same fixed and float leg parameters as _ois_pv_pure]

        Returns:
            jnp.array: Delta sensitivities [M] in USD/bp units

        Notes:
            - Uses JAX automatic differentiation for gradient computation
            - Chain rule: delta = grad(PV, DFs) @ Jacobian(DFs, rates)
            - Jacobian shared across batch (computed once during curve construction)
            - Handles prepended t=0 point via wrapper functions (same as XCCY)
            - Converts to bp units (×1e-4)

        Performance:
            - Pure JAX function: JIT compilable, vmap-compatible
            - Shared Jacobian amortizes overhead across batch
            - Per-swap gradients computed in parallel via vmap
        """
        from jax import grad
        import jax.numpy as jnp

        # Define PV function w.r.t. discount factors
        def pv_fn(dfs_var):
            return self._ois_pv_pure(
                dfs_var, times, interp_type,
                fixed_payment_times, fixed_payments, fixed_principal, fixed_leg_sign,
                float_payment_times, float_start_times, float_end_times,
                float_alphas, float_spreads, float_notionals, float_principal, float_leg_sign,
                value_time
            )

        # Use wrapper function to handle prepended t=0 (same pattern as XCCY DELTA)
        # IMPORTANT: DF(t≈0) = 1.0 is a boundary condition, NOT a curve parameter.
        # Jacobian is w.r.t. "original" DFs (excluding prepended point).

        # Extract original DFs (excluding prepended t=0 if present)
        dfs_original = dfs[1:] if times[0] < 1e-6 else dfs

        # Wrapper function that prepends t=0 before calling PV function
        def pv_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs]) if times[0] < 1e-6 else original_dfs
            return pv_fn(full_dfs)

        # Compute gradient w.r.t. ORIGINAL DFs only (excluding DF(0)=1.0)
        grad_original = grad(lambda d: jnp.squeeze(pv_original_dfs(d)))(dfs_original)

        # Apply chain rule with Jacobian
        # delta = grad(PV, DFs) @ Jacobian(DFs, rates)
        delta_rates_raw = jnp.dot(grad_original, jac)

        # Convert to bp units (rates stored in decimal, 1bp = 0.0001)
        delta = delta_rates_raw * 1e-4

        return delta

    def _ois_gamma_pure(self,
                       dfs, times, interp_type, jac, hess,
                       fixed_payment_times, fixed_payments, fixed_principal, fixed_leg_sign,
                       float_payment_times, float_start_times, float_end_times,
                       float_alphas, float_spreads, float_notionals, float_principal, float_leg_sign,
                       value_time):
        """
        Compute OIS swap GAMMA for a single swap (JAX-compatible, vmap-ready).

        Pure function for batched OIS GAMMA computation using automatic differentiation.
        Computes second-order sensitivities (GAMMA) to OIS curve rates.

        Architecture:
        1. Define PV function for OIS swap (fixed + float legs)
        2. Compute gradient and Hessian w.r.t. discount factors
        3. Apply chain rule: gamma = jac^T @ hess(PV, DFs) @ jac + sum(grad * hess_curve)
        4. Convert to bp² units

        Args:
            dfs: Discount factors [N]
            times: Curve times [N]
            interp_type: Interpolation method
            jac: Jacobian d(DFs)/d(rates) [N, M] - pre-computed from curve construction
            hess: Hessian d²(DFs)/d(rates)² [N, M] or [N, M, M] - diagonal or full

            fixed_payment_times: Fixed leg payment times [M_fixed]
            fixed_payments: Fixed leg payment amounts [M_fixed]
            fixed_principal: Final principal for fixed leg (scalar)
            fixed_leg_sign: +1 (receive fixed) or -1 (pay fixed)

            float_payment_times: Float leg payment times [M_float]
            float_start_times: Float leg period start times [M_float]
            float_end_times: Float leg period end times [M_float]
            float_alphas: Float leg year fractions [M_float]
            float_spreads: Float leg spreads [M_float]
            float_notionals: Float leg notionals [M_float]
            float_principal: Final principal for float leg (scalar)
            float_leg_sign: +1 (receive float) or -1 (pay float)

            value_time: Valuation time (scalar)

        Returns:
            jnp.array: Gamma matrix [M, M] in USD/bp² units

        Notes:
            - Single curve used for both discounting and projection (natural currency collateral)
            - No FX conversion needed (unlike XCCY)
            - Uses JAX automatic differentiation for Hessian computation
            - Chain rule: gamma = jac^T @ hess_pv_dfs @ jac + sum(grad * hess_curve)
            - Handles diagonal (ndim=2) or full (ndim=3) curve Hessians
            - Curve Hessian already excludes prepended t=0 row (shape matches Jacobian)
            - Converts to bp² units (×1e-8)

        Performance:
            - Pure JAX function: JIT compilable, vmap-compatible
            - Shared Jacobian/Hessian amortizes overhead across batch
            - Per-swap Hessians computed in parallel via vmap
        """
        from jax import grad, hessian
        import jax.numpy as jnp

        # Define PV function w.r.t. discount factors (reuse _ois_pv_pure)
        def pv_fn(dfs_var):
            return self._ois_pv_pure(
                dfs_var, times, interp_type,
                fixed_payment_times, fixed_payments, fixed_principal, fixed_leg_sign,
                float_payment_times, float_start_times, float_end_times,
                float_alphas, float_spreads, float_notionals, float_principal, float_leg_sign,
                value_time
            )

        # Use wrapper function to handle prepended t=0 (same pattern as DELTA/XCCY GAMMA)
        # IMPORTANT: DF(t≈0) = 1.0 is a boundary condition, NOT a curve parameter.
        # Jacobian and Hessian are w.r.t. "original" DFs (excluding prepended point).

        # Extract original DFs (excluding prepended t=0 if present)
        dfs_original = dfs[1:] if times[0] < 1e-6 else dfs

        # Wrapper function that prepends t=0 before calling PV function
        def pv_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs]) if times[0] < 1e-6 else original_dfs
            return pv_fn(full_dfs)

        # Define separate PV functions for fixed and float legs (matching sequential approach)
        def pv_fixed_fn(dfs_var):
            return self._price_fixed_leg_jax(
                dfs=dfs_var, times=times, interp_type=interp_type,
                payment_times=fixed_payment_times, payments=fixed_payments,
                principal=fixed_principal, leg_sign=fixed_leg_sign, value_time=value_time,
                notional=1.0
            )

        def pv_float_fn(dfs_var):
            return self._float_leg_jax(
                dfs=dfs_var, times=times,
                disc_interp_type=interp_type, idx_interp_type=interp_type,
                payment_times=float_payment_times,
                start_times=float_start_times, end_times=float_end_times,
                pay_alphas=float_alphas, spreads=float_spreads,
                notionals=float_notionals, principal=float_principal,
                leg_sign=float_leg_sign, value_time=value_time,
                first_fixing_rate=0.0, override_first=False,
                idx_times=None, idx_dfs=None,
                notional_exchange=False, notional_exchange_amount=0.0,
                effective_time=0.0, maturity_time=0.0
            )

        # Wrapper functions for fixed and float legs
        def pv_fixed_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs]) if times[0] < 1e-6 else original_dfs
            return pv_fixed_fn(full_dfs)

        def pv_float_original_dfs(original_dfs):
            full_dfs = jnp.concatenate([jnp.array([1.0]), original_dfs]) if times[0] < 1e-6 else original_dfs
            return pv_float_fn(full_dfs)

        # Compute gradients and Hessians separately for each leg (matching sequential)
        grad_fixed = grad(lambda d: jnp.squeeze(pv_fixed_original_dfs(d)))(dfs_original)
        hess_fixed_pv_dfs = hessian(lambda d: jnp.squeeze(pv_fixed_original_dfs(d)))(dfs_original)

        grad_float = grad(lambda d: jnp.squeeze(pv_float_original_dfs(d)))(dfs_original)
        hess_float_pv_dfs = hessian(lambda d: jnp.squeeze(pv_float_original_dfs(d)))(dfs_original)

        # Combined gradient and Hessian
        grad_original = grad_fixed + grad_float
        hess_pv_dfs = hess_fixed_pv_dfs + hess_float_pv_dfs

        # Apply chain rule: gamma = jac^T @ hess_pv_dfs @ jac + sum(grad * hess_curve)
        # term1: main chain rule (treating curve as fixed mapping)
        # term2: correction for curve Hessian (derivative of Jacobian itself)
        term1 = jac.T @ hess_pv_dfs @ jac

        # Handle diagonal or full curve Hessian
        # NOTE: Stored Hessians already exclude prepended t=0 row (shape matches Jacobian)
        if hess.ndim == 2:
            # Diagonal Hessian: shape (n_dfs, n_rates)
            # For diagonal: term2[j,k] = sum_i grad[i] * hess[i,j] if j==k, else 0
            term2_diag = jnp.dot(grad_original, hess)  # Shape: (n_rates,)
            term2 = jnp.diag(term2_diag)  # Shape: (n_rates, n_rates)
        else:
            # Full Hessian: shape (n_dfs, n_rates, n_rates)
            term2 = jnp.sum(grad_original[:, None, None] * hess, axis=0)

        gamma_matrix = term1 + term2

        # Convert to bp² units (1bp = 0.0001 → bp² = 1e-8)
        # OIS is in domestic currency, no FX conversion needed
        gamma = gamma_matrix * 1e-8

        return gamma

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

        # Use SensitivityEngine for DELTA and GAMMA computation (centralized implementation)
        from cavour.market.sensitivity import SensitivityEngine

        need_both = RequestTypes.DELTA in requests and RequestTypes.GAMMA in requests
        if need_both:
            # Compute both efficiently (shares gradient computation)
            delta, gamma = SensitivityEngine.compute_delta_gamma(
                pv_fn=pv_fn,
                dfs=dfs,
                jac=jac,
                hess_curve=hess_curve,
                swap_times=swap_times,
                currency=floating_leg_details._currency,
                curve_type=floating_leg_details._floating_index
            )
            out["delta"] = delta
            out["gamma"] = gamma
        else:
            # Compute only what's requested
            if RequestTypes.DELTA in requests:
                out["delta"] = SensitivityEngine.compute_delta(
                    pv_fn=pv_fn,
                    dfs=dfs,
                    jac=jac,
                    swap_times=swap_times,
                    currency=floating_leg_details._currency,
                    curve_type=floating_leg_details._floating_index
                )

            if RequestTypes.GAMMA in requests:
                out["gamma"] = SensitivityEngine.compute_gamma(
                    pv_fn=pv_fn,
                    dfs=dfs,
                    jac=jac,
                    hess_curve=hess_curve,
                    grad_dfs=None,  # Will be computed inside
                    swap_times=swap_times,
                    currency=floating_leg_details._currency,
                    curve_type=floating_leg_details._floating_index
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

###############################################################################
# Cash Deposit and FRA Computation Methods
###############################################################################

    def _compute_deposit(self, derivative, reqs, collateral_type=None):
        """Compute analytics for cash deposits (VALUE, DELTA, GAMMA).

        Uses the same AD-based approach as OIS legs:
        1. Define pricing function: pv_fn(dfs) = payment_amt * interp(maturity, times, dfs)
        2. For DELTA: grad_dfs = grad(pv_fn)(dfs), then delta = grad_dfs @ jac
        3. For GAMMA: hess_dfs = hessian(pv_fn)(dfs), then apply chain rule

        Args:
            derivative: CashDeposit instance
            reqs: Set of RequestTypes (VALUE, DELTA, GAMMA)
            collateral_type: Collateral type (future extensibility)

        Returns:
            AnalyticsResult with value, risk (delta), and gamma
        """
        # Get the curve name from the deposit's floating index
        curve_name = derivative._floating_index.name
        ir_model = getattr(self.model.curves, curve_name)

        # Get cached curve data (times, DFs, Jacobian, Hessian)
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

        # Compute maturity time in years
        from cavour.utils.helpers import times_from_dates
        maturity_time = times_from_dates(
            derivative._maturity_dt,
            ir_model._value_dt,
            derivative._dc_type
        )

        # Get interpolator
        from cavour.market.curves.interpolator_ad import InterpolatorAd
        interpolator = InterpolatorAd(ir_model._interp_type)
        interpolator.fit(times, dfs)

        # Define pricing function: PV = payment_amt * DF(maturity) / DF(value_dt) - notional
        # For value_dt = effective_dt (curve construction), DF(value_dt) = 1.0
        def pv_fn(dfs_array):
            """Pricing function for JAX autodiff."""
            df_mat = interpolator.simple_interpolate(maturity_time, times, dfs_array, ir_model._interp_type.value)
            pv_payment = derivative._payment_amt * df_mat  # Assuming value_dt = effective_dt, so df_value = 1
            return pv_payment - derivative._notional

        # Compute VALUE
        value = None
        if RequestTypes.VALUE in reqs:
            val = pv_fn(dfs)
            val_scalar = float(jnp.atleast_1d(val).item() if jnp.ndim(val) == 0 else val.squeeze())
            value = Valuation(amount=val_scalar, currency=derivative._currency)

        # Use SensitivityEngine for DELTA and GAMMA computation (centralized implementation)
        from cavour.market.sensitivity import SensitivityEngine

        delta = None
        gamma = None

        need_both = RequestTypes.DELTA in reqs and RequestTypes.GAMMA in reqs
        if need_both:
            # Check GAMMA precondition
            if hess_curve is None:
                raise LibError("GAMMA requested but curve was not built with compute_gamma=True")

            # Compute both efficiently (shares gradient computation)
            delta, gamma = SensitivityEngine.compute_delta_gamma(
                pv_fn=pv_fn,
                dfs=dfs,
                jac=jac,
                hess_curve=hess_curve,
                swap_times=ir_model.swap_times,
                currency=derivative._currency,
                curve_type=derivative._floating_index
            )
        else:
            # Compute only what's requested
            if RequestTypes.DELTA in reqs:
                delta = SensitivityEngine.compute_delta(
                    pv_fn=pv_fn,
                    dfs=dfs,
                    jac=jac,
                    swap_times=ir_model.swap_times,
                    currency=derivative._currency,
                    curve_type=derivative._floating_index
                )

            if RequestTypes.GAMMA in reqs:
                if hess_curve is None:
                    raise LibError("GAMMA requested but curve was not built with compute_gamma=True")

                gamma = SensitivityEngine.compute_gamma(
                    pv_fn=pv_fn,
                    dfs=dfs,
                    jac=jac,
                    hess_curve=hess_curve,
                    grad_dfs=None,  # Will be computed inside
                    swap_times=ir_model.swap_times,
                    currency=derivative._currency,
                    curve_type=derivative._floating_index
                )

        return AnalyticsResult(value=value, risk=delta, gamma=gamma, cashflows=None)

    def _compute_stir_future(self, derivative, reqs, collateral_type=None):
        """Compute analytics for IR Futures (VALUE, DELTA, GAMMA).

        Uses the same AD-based approach as FRAs:
        1. Define pricing function: pv_fn(dfs) with two interpolations (start, end)
        2. For DELTA: grad_dfs = grad(pv_fn)(dfs), then delta = grad_dfs @ jac
        3. For GAMMA: hess_dfs = hessian(pv_fn)(dfs), then apply chain rule

        Futures pricing formula (no discounting - daily mark-to-market):
        PV = notional × (df_start/df_end - 1 - futures_rate × year_frac)

        Args:
            derivative: IRFuture instance
            reqs: Set of RequestTypes (VALUE, DELTA, GAMMA)
            collateral_type: Collateral type (future extensibility)

        Returns:
            AnalyticsResult with value, risk (delta), and gamma
        """
        # Get the curve name from the future's floating index
        curve_name = derivative._floating_index.name
        ir_model = getattr(self.model.curves, curve_name)

        # Get cached curve data (times, DFs, Jacobian, Hessian)
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

        # Compute two time points in years (accrual start and end)
        from cavour.utils.helpers import times_from_dates
        start_time = times_from_dates(
            derivative._accrual_start_dt, ir_model._value_dt, derivative._dc_type
        )
        end_time = times_from_dates(
            derivative._accrual_end_dt, ir_model._value_dt, derivative._dc_type
        )

        # Get interpolator
        from cavour.market.curves.interpolator_ad import InterpolatorAd
        interpolator = InterpolatorAd(ir_model._interp_type)
        interpolator.fit(times, dfs)

        # Future parameters
        notional = derivative._notional
        year_frac = derivative._year_frac
        futures_rate = derivative._forward_rate  # Implied forward rate from futures price

        # Define pricing function: PV = notional × (df_start/df_end - 1 - futures_rate × year_frac)
        # Note: NO discounting for futures (daily mark-to-market with cash settlement)
        def pv_fn(dfs_array):
            """Pricing function for JAX autodiff."""
            df_start = interpolator.simple_interpolate(start_time, times, dfs_array, ir_model._interp_type.value)
            df_end = interpolator.simple_interpolate(end_time, times, dfs_array, ir_model._interp_type.value)

            # Rate differential: (df_start/df_end - 1)/year_frac - futures_rate
            # Simplifies to: (df_start/df_end - 1 - futures_rate × year_frac)
            return notional * (df_start / df_end - 1.0 - futures_rate * year_frac)

        # Compute VALUE
        value = None
        if RequestTypes.VALUE in reqs:
            val = pv_fn(dfs)
            val_scalar = float(jnp.atleast_1d(val).item() if jnp.ndim(val) == 0 else val.squeeze())
            value = Valuation(amount=val_scalar, currency=derivative._currency)

        # Use SensitivityEngine for DELTA and GAMMA computation (centralized implementation)
        from cavour.market.sensitivity import SensitivityEngine

        delta = None
        gamma = None

        need_both = RequestTypes.DELTA in reqs and RequestTypes.GAMMA in reqs
        if need_both:
            # Check GAMMA precondition
            if hess_curve is None:
                raise LibError("GAMMA requested but curve was not built with compute_gamma=True")

            # Compute both efficiently (shares gradient computation)
            delta, gamma = SensitivityEngine.compute_delta_gamma(
                pv_fn=pv_fn,
                dfs=dfs,
                jac=jac,
                hess_curve=hess_curve,
                swap_times=ir_model.swap_times,
                currency=derivative._currency,
                curve_type=derivative._floating_index
            )
        else:
            # Compute only what's requested
            if RequestTypes.DELTA in reqs:
                delta = SensitivityEngine.compute_delta(
                    pv_fn=pv_fn,
                    dfs=dfs,
                    jac=jac,
                    swap_times=ir_model.swap_times,
                    currency=derivative._currency,
                    curve_type=derivative._floating_index
                )

            if RequestTypes.GAMMA in reqs:
                if hess_curve is None:
                    raise LibError("GAMMA requested but curve was not built with compute_gamma=True")

                gamma = SensitivityEngine.compute_gamma(
                    pv_fn=pv_fn,
                    dfs=dfs,
                    jac=jac,
                    hess_curve=hess_curve,
                    grad_dfs=None,  # Will be computed inside
                    swap_times=ir_model.swap_times,
                    currency=derivative._currency,
                    curve_type=derivative._floating_index
                )

        return AnalyticsResult(value=value, risk=delta, gamma=gamma, cashflows=None)

    def _compute_fra(self, derivative, reqs, collateral_type=None):
        """Compute analytics for FRAs (VALUE, DELTA, GAMMA).

        Uses the same AD-based approach as deposits and OIS legs:
        1. Define pricing function: pv_fn(dfs) with three interpolations
        2. For DELTA: grad_dfs = grad(pv_fn)(dfs), then delta = grad_dfs @ jac
        3. For GAMMA: hess_dfs = hessian(pv_fn)(dfs), then apply chain rule

        FRA pricing formula:
        PV = notional × (df_start/df_end - 1 - fra_rate × year_frac) × df_settlement

        Args:
            derivative: FRA instance
            reqs: Set of RequestTypes (VALUE, DELTA, GAMMA)
            collateral_type: Collateral type (future extensibility)

        Returns:
            AnalyticsResult with value, risk (delta), and gamma
        """
        # Get the curve name from the FRA's floating index
        curve_name = derivative._floating_index.name
        ir_model = getattr(self.model.curves, curve_name)

        # Get cached curve data (times, DFs, Jacobian, Hessian)
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

        # Compute three time points in years
        from cavour.utils.helpers import times_from_dates
        start_time = times_from_dates(
            derivative._start_dt, ir_model._value_dt, derivative._dc_type
        )
        end_time = times_from_dates(
            derivative._end_dt, ir_model._value_dt, derivative._dc_type
        )
        settlement_time = times_from_dates(
            derivative._settlement_dt, ir_model._value_dt, derivative._dc_type
        )

        # Get interpolator
        from cavour.market.curves.interpolator_ad import InterpolatorAd
        interpolator = InterpolatorAd(ir_model._interp_type)
        interpolator.fit(times, dfs)

        # FRA parameters
        notional = derivative._notional
        year_frac = derivative._year_frac
        fra_rate = derivative._fra_rate

        # Define pricing function: PV = payoff × df_settlement
        # payoff = notional × (df_start/df_end - 1 - fra_rate × year_frac)
        def pv_fn(dfs_array):
            """Pricing function for JAX autodiff."""
            df_start = interpolator.simple_interpolate(start_time, times, dfs_array, ir_model._interp_type.value)
            df_end = interpolator.simple_interpolate(end_time, times, dfs_array, ir_model._interp_type.value)
            df_settlement = interpolator.simple_interpolate(settlement_time, times, dfs_array, ir_model._interp_type.value)

            payoff = notional * (df_start / df_end - 1.0 - fra_rate * year_frac)
            return payoff * df_settlement

        # Compute VALUE
        value = None
        if RequestTypes.VALUE in reqs:
            val = pv_fn(dfs)
            val_scalar = float(jnp.atleast_1d(val).item() if jnp.ndim(val) == 0 else val.squeeze())
            value = Valuation(amount=val_scalar, currency=derivative._currency)

        # Use SensitivityEngine for DELTA and GAMMA computation (centralized implementation)
        from cavour.market.sensitivity import SensitivityEngine

        delta = None
        gamma = None

        need_both = RequestTypes.DELTA in reqs and RequestTypes.GAMMA in reqs
        if need_both:
            # Check GAMMA precondition
            if hess_curve is None:
                raise LibError("GAMMA requested but curve was not built with compute_gamma=True")

            # Compute both efficiently (shares gradient computation)
            delta, gamma = SensitivityEngine.compute_delta_gamma(
                pv_fn=pv_fn,
                dfs=dfs,
                jac=jac,
                hess_curve=hess_curve,
                swap_times=ir_model.swap_times,
                currency=derivative._currency,
                curve_type=derivative._floating_index
            )
        else:
            # Compute only what's requested
            if RequestTypes.DELTA in reqs:
                delta = SensitivityEngine.compute_delta(
                    pv_fn=pv_fn,
                    dfs=dfs,
                    jac=jac,
                    swap_times=ir_model.swap_times,
                    currency=derivative._currency,
                    curve_type=derivative._floating_index
                )

            if RequestTypes.GAMMA in reqs:
                if hess_curve is None:
                    raise LibError("GAMMA requested but curve was not built with compute_gamma=True")

                gamma = SensitivityEngine.compute_gamma(
                    pv_fn=pv_fn,
                    dfs=dfs,
                    jac=jac,
                    hess_curve=hess_curve,
                    grad_dfs=None,  # Will be computed inside
                    swap_times=ir_model.swap_times,
                    currency=derivative._currency,
                    curve_type=derivative._floating_index
                )

        return AnalyticsResult(value=value, risk=delta, gamma=gamma, cashflows=None)

    def _compute_stir_future(self, derivative, reqs, collateral_type=None):
        """Compute analytics for IR Futures (VALUE, DELTA, GAMMA).

        Uses the same AD-based approach as FRAs:
        1. Define pricing function: pv_fn(dfs) with two interpolations (start, end)
        2. For DELTA: grad_dfs = grad(pv_fn)(dfs), then delta = grad_dfs @ jac
        3. For GAMMA: hess_dfs = hessian(pv_fn)(dfs), then apply chain rule

        Futures pricing formula (no discounting - daily mark-to-market):
        PV = notional × (df_start/df_end - 1 - futures_rate × year_frac)

        Args:
            derivative: IRFuture instance
            reqs: Set of RequestTypes (VALUE, DELTA, GAMMA)
            collateral_type: Collateral type (future extensibility)

        Returns:
            AnalyticsResult with value, risk (delta), and gamma
        """
        # Get the curve name from the future's floating index
        curve_name = derivative._floating_index.name
        ir_model = getattr(self.model.curves, curve_name)

        # Get cached curve data (times, DFs, Jacobian, Hessian)
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

        # Compute two time points in years (accrual start and end)
        from cavour.utils.helpers import times_from_dates
        start_time = times_from_dates(
            derivative._accrual_start_dt, ir_model._value_dt, derivative._dc_type
        )
        end_time = times_from_dates(
            derivative._accrual_end_dt, ir_model._value_dt, derivative._dc_type
        )

        # Get interpolator
        from cavour.market.curves.interpolator_ad import InterpolatorAd
        interpolator = InterpolatorAd(ir_model._interp_type)
        interpolator.fit(times, dfs)

        # Future parameters
        notional = derivative._notional
        year_frac = derivative._year_frac
        futures_rate = derivative._forward_rate  # Implied forward rate from futures price

        # Define pricing function: PV = notional × (df_start/df_end - 1 - futures_rate × year_frac)
        # Note: NO discounting for futures (daily mark-to-market with cash settlement)
        def pv_fn(dfs_array):
            """Pricing function for JAX autodiff."""
            df_start = interpolator.simple_interpolate(start_time, times, dfs_array, ir_model._interp_type.value)
            df_end = interpolator.simple_interpolate(end_time, times, dfs_array, ir_model._interp_type.value)

            # Rate differential: (df_start/df_end - 1)/year_frac - futures_rate
            # Simplifies to: (df_start/df_end - 1 - futures_rate × year_frac)
            return notional * (df_start / df_end - 1.0 - futures_rate * year_frac)

        # Compute VALUE
        value = None
        if RequestTypes.VALUE in reqs:
            val = pv_fn(dfs)
            val_scalar = float(jnp.atleast_1d(val).item() if jnp.ndim(val) == 0 else val.squeeze())
            value = Valuation(amount=val_scalar, currency=derivative._currency)

        # Use SensitivityEngine for DELTA and GAMMA computation (centralized implementation)
        from cavour.market.sensitivity import SensitivityEngine

        delta = None
        gamma = None

        need_both = RequestTypes.DELTA in reqs and RequestTypes.GAMMA in reqs
        if need_both:
            # Check GAMMA precondition
            if hess_curve is None:
                raise LibError("GAMMA requested but curve was not built with compute_gamma=True")

            # Compute both efficiently (shares gradient computation)
            delta, gamma = SensitivityEngine.compute_delta_gamma(
                pv_fn=pv_fn,
                dfs=dfs,
                jac=jac,
                hess_curve=hess_curve,
                swap_times=ir_model.swap_times,
                currency=derivative._currency,
                curve_type=derivative._floating_index
            )
        else:
            # Compute only what's requested
            if RequestTypes.DELTA in reqs:
                delta = SensitivityEngine.compute_delta(
                    pv_fn=pv_fn,
                    dfs=dfs,
                    jac=jac,
                    swap_times=ir_model.swap_times,
                    currency=derivative._currency,
                    curve_type=derivative._floating_index
                )

            if RequestTypes.GAMMA in reqs:
                if hess_curve is None:
                    raise LibError("GAMMA requested but curve was not built with compute_gamma=True")

                gamma = SensitivityEngine.compute_gamma(
                    pv_fn=pv_fn,
                    dfs=dfs,
                    jac=jac,
                    hess_curve=hess_curve,
                    grad_dfs=None,  # Will be computed inside
                    swap_times=ir_model.swap_times,
                    currency=derivative._currency,
                    curve_type=derivative._floating_index
                )

        return AnalyticsResult(value=value, risk=delta, gamma=gamma, cashflows=None)