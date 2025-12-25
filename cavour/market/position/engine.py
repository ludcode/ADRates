"Valuation Engine"

import numpy as np
import jax.numpy as jnp
from jax import lax, jit, grad, hessian, jacrev, linearize
from functools import partial
from typing import Sequence, Any, Dict

from cavour.market.curves.interpolator import *
from cavour.utils.helpers import to_tenor, times_from_dates
from cavour.utils.date import Date
from cavour.market.curves.interpolator_ad import InterpolatorAd
from cavour.requests.results import Valuation, Gamma, Delta, AnalyticsResult, Risk
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

    def compute(self, derivative, request_list):
        """Return analytics for the given derivative and requested measures."""
        reqs = set(request_list)

        # Route XCCY swaps to separate handler
        if derivative.derivative_type == InstrumentTypes.XCCY_SWAP:
            return self._compute_xccy(derivative, reqs)

        # Route OIS swaps to existing logic
        if derivative.derivative_type != InstrumentTypes.OIS_SWAP:
            raise LibError(f"{derivative.derivative_type} not yet implemented")

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

        #risk = Risk([delta]) if delta is not None else None

        return AnalyticsResult(value=value, risk=delta, gamma=gamma)

    def _compute_xccy(self, derivative, reqs):
        """Compute analytics for cross-currency swaps (VALUE, DELTA, GAMMA).

        Handles XccyFixFloat, XccyBasisSwap, and XccyFixFix swaps.
        Currently supports VALUE only.

        Args:
            derivative: XCCY swap instance (XccyFixFloat, XccyBasisSwap, or XccyFixFix)
            reqs: Set of RequestTypes (VALUE, DELTA, GAMMA)

        Returns:
            AnalyticsResult with value, risk (multi-curve Delta), and gamma
        """
        # Get spot FX rate from model
        try:
            foreign_code = derivative._foreign_currency.name
            domestic_code = derivative._domestic_currency.name
            fx_pair = f"{foreign_code}{domestic_code}"  # e.g., "USDGBP"

            if fx_pair not in self.model._fx_params_dict:
                raise LibError(f"FX rate {fx_pair} not found in model. Available: {list(self.model._fx_params_dict.keys())}")

            spot_fx = self.model._fx_params_dict[fx_pair]["price"]
        except LibError:
            raise

        # Get curves from model
        domestic_model = getattr(self.model.curves, derivative._domestic_floating_index.name)
        foreign_model = getattr(self.model.curves, derivative._foreign_floating_index.name)

        # Detect leg types to route appropriately
        is_domestic_fixed = isinstance(derivative._domestic_leg, SwapFixedLeg)
        is_foreign_fixed = isinstance(derivative._foreign_leg, SwapFixedLeg)

        # Compute domestic leg analytics
        if is_domestic_fixed:
            # For XCCY swaps, use direct leg valuation to match test behavior
            domestic_leg_value = derivative._domestic_leg.value(
                value_dt=domestic_model._value_dt,
                discount_curve=domestic_model
            )
            domestic_analytics = {
                "value": Valuation(amount=domestic_leg_value, currency=derivative._domestic_currency)
            }
        else:
            # For XCCY floating legs with notional_exchange=True, use direct valuation
            if derivative._domestic_leg._notional_exchange:
                domestic_leg_value = derivative._domestic_leg.value(
                    value_dt=domestic_model._value_dt,
                    discount_curve=domestic_model,
                    index_curve=domestic_model,
                    first_fixing_rate=None
                )
                domestic_analytics = {
                    "value": Valuation(amount=domestic_leg_value, currency=derivative._domestic_currency)
                }
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
            # For XCCY swaps, use direct leg valuation to match test behavior
            foreign_leg_value = derivative._foreign_leg.value(
                value_dt=foreign_model._value_dt,
                discount_curve=foreign_model
            )
            foreign_analytics = {
                "value": Valuation(amount=foreign_leg_value, currency=derivative._foreign_currency)
            }
        else:
            # For XCCY floating legs with notional_exchange=True, use direct valuation
            # because _float_leg_analytics doesn't handle notional exchanges correctly
            # (end notional is added to _payments[-1], but _float_leg_jax recalculates from scratch)
            if derivative._foreign_leg._notional_exchange:
                # Call leg.value() directly
                foreign_leg_value = derivative._foreign_leg.value(
                    value_dt=foreign_model._value_dt,
                    discount_curve=foreign_model,  # Use foreign curve for both discount and index
                    index_curve=foreign_model,
                    first_fixing_rate=None
                )
                foreign_analytics = {
                    "value": Valuation(amount=foreign_leg_value, currency=derivative._foreign_currency)
                }
            else:
                foreign_analytics = self._float_leg_analytics(
                    foreign_model.swap_rates,
                    foreign_model.swap_times,
                    foreign_model.year_fracs,
                    derivative._foreign_leg,
                    foreign_model._value_dt,
                    foreign_model._interp_type,  # discount curve
                    foreign_model._interp_type,  # index curve
                    None,  # first_fixing_rate
                    reqs,
                )

        # Add notional exchanges for fixed legs (they don't include them by default)
        if RequestTypes.VALUE in reqs:
            # Domestic leg notional exchanges (if fixed)
            if is_domestic_fixed:
                domestic_value = domestic_analytics["value"].amount
                effective_dt = derivative._effective_dt
                maturity_dt = derivative._maturity_dt
                domestic_leg_type = derivative._domestic_leg_type

                # Start exchange: -notional at effective date (outflow)
                if effective_dt >= domestic_model._value_dt:
                    df_start = domestic_model.df(effective_dt)
                    start_exchange_pv = -derivative._domestic_notional * df_start
                    # Apply leg type sign
                    if domestic_leg_type == SwapTypes.RECEIVE:
                        domestic_value += start_exchange_pv
                    else:
                        domestic_value -= start_exchange_pv

                # End exchange: +notional at maturity date (inflow)
                if maturity_dt >= domestic_model._value_dt:
                    df_end = domestic_model.df(maturity_dt)
                    end_exchange_pv = derivative._domestic_notional * df_end
                    # Apply leg type sign
                    if domestic_leg_type == SwapTypes.RECEIVE:
                        domestic_value += end_exchange_pv
                    else:
                        domestic_value -= end_exchange_pv

                # Update domestic analytics with notional exchanges
                domestic_analytics["value"] = Valuation(
                    amount=domestic_value,
                    currency=derivative._domestic_currency
                )

            # Foreign leg notional exchanges (if fixed)
            # Note: Floating legs already include notional exchanges via notional_exchange=True
            if is_foreign_fixed:
                foreign_value = foreign_analytics["value"].amount
                effective_dt = derivative._effective_dt
                maturity_dt = derivative._maturity_dt
                foreign_leg_type = derivative._foreign_leg._leg_type

                # Start exchange: -notional at effective date (outflow)
                if effective_dt >= foreign_model._value_dt:
                    df_start = foreign_model.df(effective_dt)
                    start_exchange_pv = -derivative._foreign_notional * df_start
                    # Apply leg type sign
                    if foreign_leg_type == SwapTypes.RECEIVE:
                        foreign_value += start_exchange_pv
                    else:
                        foreign_value -= start_exchange_pv

                # End exchange: +notional at maturity date (inflow)
                if maturity_dt >= foreign_model._value_dt:
                    df_end = foreign_model.df(maturity_dt)
                    end_exchange_pv = derivative._foreign_notional * df_end
                    # Apply leg type sign
                    if foreign_leg_type == SwapTypes.RECEIVE:
                        foreign_value += end_exchange_pv
                    else:
                        foreign_value -= end_exchange_pv

                # Update foreign analytics with notional exchanges
                foreign_analytics["value"] = Valuation(
                    amount=foreign_value,
                    currency=derivative._foreign_currency
                )

        # Combine results
        value = None
        if RequestTypes.VALUE in reqs:
            domestic_value = domestic_analytics["value"].amount
            foreign_value = foreign_analytics["value"].amount
            # Total PV = domestic PV + spot_FX * foreign PV (converted to domestic currency)
            total_value = domestic_value + spot_fx * foreign_value
            value = Valuation(amount=total_value, currency=derivative._domestic_currency)

        # DELTA and GAMMA not yet implemented for XCCY
        delta = None
        gamma = None

        return AnalyticsResult(value=value, risk=delta, gamma=gamma)

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
                - all_maturities: All unique intermediate times including swap maturities
                - all_dfs: Discount factors at all intermediate times

        Implementation:
            1. Pre-expands all intermediate cashflow points from all swaps
            2. Deduplicates using rounded maturity keys (2 decimal places)
            3. Builds dependency graph via prev_idx mapping
            4. Sequential bootstrap via lax.scan
            5. Returns dense grid for interpolation (not just swap maturities)

        Note:
            Each intermediate point inherits its parent swap's rate, matching
            the recursive version's behavior. Rounding to 2 decimals supports
            quarterly swaps (0.25) and is used only for dictionary key matching,
            not for actual computations.
        """

        # 1) Pre-expand ALL intermediate points (not just swap maturities)
        points = []
        for i, (rate, fracs) in enumerate(zip(swap_rates, year_fracs)):
            cumsum = 0.0
            for j, frac in enumerate(fracs):
                prev_cum = cumsum
                cumsum += frac
                points.append({
                    'maturity': cumsum,                      # EXACT value for computations
                    'maturity_key': round(cumsum, 4),       # Rounded key for matching (4 decimals to avoid collisions)
                    'acc': frac,
                    'prev_mat': prev_cum,                    # EXACT previous maturity
                    'prev_key': round(prev_cum, 4) if j > 0 else None,  # Rounded key (4 decimals)
                    'rate': rate,                            # Parent swap's rate
                    'is_final': (j == len(fracs) - 1),      # Is this the swap's final maturity?
                    'swap_idx': i
                })

        # 2) Sort by exact maturity
        sorted_points = sorted(points, key=lambda x: x['maturity'])

        # 3) Deduplicate using rounded keys (keep first occurrence like the recursive version)
        seen_keys = {}
        unique_points = []
        for p in sorted_points:
            key = p['maturity_key']
            if key not in seen_keys:
                seen_keys[key] = len(unique_points)
                unique_points.append(p)

        # 4) Build maturity_key → index mapping (for prev_idx lookup)
        maturity_lookup = {}  # rounded_key → index in unique_points
        for idx, p in enumerate(unique_points):
            maturity_lookup[p['maturity_key']] = idx

        # 5) Build prev_idx for each point using rounded keys
        for p in unique_points:
            if p['prev_key'] is None:
                p['prev_idx'] = -1
            else:
                p['prev_idx'] = maturity_lookup.get(p['prev_key'], -1)

        # 6) Convert to JAX arrays
        n_points = len(unique_points)
        rates = jnp.array([p['rate'] for p in unique_points])
        accs = jnp.array([p['acc'] for p in unique_points])
        prev_idxs = jnp.array([p['prev_idx'] for p in unique_points], dtype=jnp.int32)

        # 7) JAX-friendly scan through all points
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

        # 8) Run the scan
        init_pv01 = jnp.zeros(n_points)
        idxs = jnp.arange(n_points)
        _, all_dfs = lax.scan(step, init_pv01, (idxs, rates, accs, prev_idxs))

        # 9) Return ALL unique intermediate points (not just swap maturities)
        # This matches ois_curve.py which stores all intermediate DFs for interpolation
        all_maturities = jnp.array([p['maturity'] for p in unique_points])
        return all_maturities, all_dfs

    def _cached_curve(self, key, swap_rates, swap_times, year_fracs, interp_type):
        """Bootstrap the curve once and cache DFS, Jacobian and Hessian."""
        cache = self._curve_cache.get(key)
        if cache is not None:
            return cache

        # Build curve to get both times and DFs (including all intermediate points)
        rates = jnp.array(swap_rates)
        times, dfs = self.build_curve_ad(rates, swap_times, year_fracs)

        # For AD, we need DFs as a function of rates only (times are constant)
        build_dfs = lambda r: self.build_curve_ad(r, swap_times, year_fracs)[1]
        jac = jacrev(build_dfs)(rates)
        hess = hessian(build_dfs)(rates)

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
                    override_first               # scalar
                    ):
        disc_interp = InterpolatorAd(disc_interp_type)
        idx_interp  = InterpolatorAd(idx_interp_type)

        df_val   = jnp.atleast_1d(disc_interp.simple_interpolate(value_time, times, dfs, disc_interp_type.value))
        df_start = jnp.atleast_1d(idx_interp.simple_interpolate(start_times, times, dfs, idx_interp_type.value))
        df_end   = jnp.atleast_1d(idx_interp.simple_interpolate(end_times, times, dfs, idx_interp_type.value))

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

        # i) aggregate and apply sign
        leg_pv     = jnp.sum(pv_coupons, axis=-1) + pv_prin                      # [...]
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

        curve_key = tuple(swap_times)
        cache = self._cached_curve(
            curve_key, swap_rates, swap_times, year_fracs, discount_curve_type
        )
        times = cache["times"]
        dfs = cache["dfs"]
        jac = cache["jac"]
        hess_curve = cache["hess"]

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