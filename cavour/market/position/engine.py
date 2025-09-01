"Valuation Engine"

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, jit, grad, hessian, jacrev, linearize
from functools import partial
from typing import Sequence, Any, Dict

from cavour.utils.global_types import InterpTypes
from cavour.utils.helpers import to_tenor, times_from_dates
from cavour.utils.date import Date
from cavour.utils.error import LibError
from cavour.market.curves.interpolator_ad import InterpolatorAd
from cavour.requests.results import Valuation, Gamma, Delta, AnalyticsResult
from cavour.utils.global_types import (SwapTypes, 
                                   InstrumentTypes, 
                                   RequestTypes,
                                   CurveTypes)
from cavour.utils.currency import CurrencyTypes



class Engine:
    def __init__(self,
                 model):

        self.model = model
        # cache bootstrapped curves keyed by curve name
        self._curve_cache: Dict[Any, Dict[str, Any]] = {}

    def compute(self, derivative, request_list):
        """Return analytics for the given derivative and requested measures."""
        reqs = set(request_list)

        if derivative.derivative_type != InstrumentTypes.OIS_SWAP:
            raise LibError(f"{derivative.derivative_type} not yet implemented")

        ir_model = getattr(self.model.curves, derivative._floating_index.name)

        # Batch gradient computations by passing all risk requests together
        # This allows the caching mechanism to compute gradients once and reuse them
        fixed = self._fixed_leg_analytics(
            ir_model.swap_rates,
            ir_model.swap_times,
            ir_model.year_fracs,
            derivative._fixed_leg,
            ir_model._value_dt,
            ir_model._interp_type,
            reqs,  # Pass all requests to enable batched gradient computation
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
            reqs,  # Pass all requests to enable batched gradient computation
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
    

    @partial(jit, static_argnums=(0,))
    def build_curve_ad(self,
                    swap_rates: list[float],
                    swap_times: list[float],
                    year_fracs: list[list[float]]
                    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Bootstraps an OIS curve via par-swap rates.
        Inputs as Python lists; outputs as JAX arrays.
        """

        # 1) Convert inputs to JAX arrays
        rates = jnp.array(swap_rates)    # shape (N,)
        times = jnp.array(swap_times)    # shape (N,)
        N     = rates.shape[0]

        # 2) Precompute prev_idx & acc_last - keep EXACT same logic but JAX-traceable
        times_rounded_jax = jnp.round(times, 1)  # JAX version of rounding
        
        # Pre-process year_fracs data structures for JAX compatibility
        max_len = max(len(fr) for fr in year_fracs)
        year_fracs_lengths = jnp.array([len(fr) for fr in year_fracs])
        
        # Pad year_fracs to enable JAX operations (pad with zeros - won't affect sums)
        year_fracs_padded = jnp.array([
            fr + [0.0] * (max_len - len(fr)) for fr in year_fracs
        ])
        
        def compute_prev_idx_acc_last(i):
            fr_len = year_fracs_lengths[i]
            
            # Case 1: if len(fr) == 1 → prev_idx = -1, acc_last = fr[0]
            case1_prev_idx = -1
            case1_acc_last = year_fracs_padded[i, 0]
            
            # Case 2: len(fr) > 1 → compute sum(fr[:-1]) and find index
            # Sum all but last: sum(fr[:-1]) = sum first (fr_len-1) elements
            cumsum = jnp.cumsum(year_fracs_padded[i])
            last_pay_unrounded = jnp.where(fr_len > 1, cumsum[fr_len-2], 0.0)
            last_pay_rounded = jnp.round(last_pay_unrounded, 1)
            
            # Check if last_pay_rounded in times_rounded (exact match search)
            matches = jnp.isclose(times_rounded_jax, last_pay_rounded, atol=1e-10)
            has_exact_match = jnp.any(matches)
            exact_match_idx = jnp.argmax(matches)  # index of first match
            
            # Fallback: nearest tenor (minimum absolute difference)
            diffs = jnp.abs(times - last_pay_rounded)
            nearest_idx = jnp.argmin(diffs)
            
            # Select index: use exact match if found, otherwise nearest
            case2_prev_idx = jnp.where(has_exact_match, exact_match_idx, nearest_idx)
            case2_acc_last = year_fracs_padded[i, fr_len-1]  # last element: fr[-1]
            
            # Final selection between case 1 and case 2
            prev_idx_val = jnp.where(fr_len == 1, case1_prev_idx, case2_prev_idx)
            acc_last_val = jnp.where(fr_len == 1, case1_acc_last, case2_acc_last)
            
            return prev_idx_val, acc_last_val
        
        # Apply to all indices using vmap for efficiency
        results = jax.vmap(compute_prev_idx_acc_last)(jnp.arange(N))
        prev_idx = results[0].astype(jnp.int32)  # shape (N,)
        acc_last = results[1]                    # shape (N,)


        # 3) JAX-friendly scan step
        def step(pv01_arr, inputs):
            i, r = inputs
            pi    = prev_idx[i]
            prev  = jnp.where(pi < 0, 0.0, pv01_arr[pi])
            a     = acc_last[i]

            df_i = jnp.where(
                pi < 0,
                1.0 / (1.0 + r * a),
                (1.0 - r * prev) / (1.0 + r * a)
            )

            pv01_i   = prev + a * df_i
            new_pv01 = pv01_arr.at[i].set(pv01_i)
            return new_pv01, df_i

        # 4) Run the scan
        init_pv01 = jnp.zeros_like(rates)
        idxs       = jnp.arange(N)
        _, dfs_arr = lax.scan(step, init_pv01, (idxs, rates))

        # 5) Return JAX arrays
        return times, dfs_arr

    def _cached_curve(self, key, swap_rates, swap_times, year_fracs, interp_type):
        """Bootstrap the curve once and cache DFS, Jacobian and Hessian."""
        cache = self._curve_cache.get(key)
        if cache is not None:
            return cache

        # Create JIT-compiled versions for massive speedup
        @jit
        def build(r):
            return self.build_curve_ad(r, swap_times, year_fracs)[1]
        
        @jit  
        def jac_fn(r):
            return jacrev(build)(r)
            
        @jit
        def hess_fn(r):
            return hessian(build)(r)
        
        rates = jnp.array(swap_rates)
        times = jnp.array(swap_times)
        
        # These will compile on first call, then be blazing fast
        dfs = build(rates)
        jac = jac_fn(rates)  
        hess = hess_fn(rates)
        cache = {
            "times": times,
            "dfs": dfs,
            "jac": jac,
            "hess": hess,
            "gradient_cache": {},  # Cache for leg-specific gradients
        }
        self._curve_cache[key] = cache
        return cache
    
    def _get_cached_gradients(self, cache, pv_fn, dfs, leg_signature, compute_hessian=False):
        """Get or compute gradients for a pricing function and cache them."""
        gradient_cache = cache["gradient_cache"]
        
        if leg_signature in gradient_cache:
            cached_grads = gradient_cache[leg_signature]
            grad_dfs = cached_grads["grad_dfs"]
            hess_dfs = cached_grads.get("hess_dfs")
            
            # If hessian is requested but not cached, compute it now
            if compute_hessian and hess_dfs is None:
                hess_dfs = hessian(lambda d: pv_fn(d))(dfs)
                cached_grads["hess_dfs"] = hess_dfs
                
        else:
            # Compute gradients (always compute both if hessian is needed for efficiency)
            grad_dfs = grad(lambda d: pv_fn(d))(dfs)
            hess_dfs = hessian(lambda d: pv_fn(d))(dfs) if compute_hessian else None
            
            # Cache the results
            gradient_cache[leg_signature] = {
                "grad_dfs": grad_dfs,
                "hess_dfs": hess_dfs
            }
        
        return grad_dfs, hess_dfs
        
    @partial(jit, static_argnums=(0, 3))  # self and interp_type are static
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
        df_val   = interp.simple_interpolate(value_time, times, dfs, interp_type.value)
        df_pmts  = interp.simple_interpolate(payment_times, times, dfs, interp_type.value)

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
            out["value"] = Valuation(amount=float(val), currency=fixed_leg_details._currency)

        need_grad = RequestTypes.DELTA in requests or RequestTypes.GAMMA in requests
        compute_hessian = RequestTypes.GAMMA in requests
        
        if need_grad:
            # Create a unique signature for this fixed leg
            leg_signature = ("fixed", hash(tuple(payment_times.tolist() + payments.tolist())), 
                           principal, notional, leg_sign, value_time)
            
            grad_dfs, hess_dfs = self._get_cached_gradients(
                cache, pv_fn, dfs, leg_signature, compute_hessian
            )

        if RequestTypes.DELTA in requests:
            sensitivities = jnp.dot(grad_dfs, jac)
            sensies = [float(x) * 1e-4 for x in sensitivities]
            out["delta"] = Delta(
                risk_ladder=sensies,
                tenors=to_tenor(swap_times),
                currency=fixed_leg_details._currency,
                curve_type=CurveTypes.GBP_OIS_SONIA,
            )

        if RequestTypes.GAMMA in requests:
            term1 = jac.T @ hess_dfs @ jac
            term2 = jnp.sum(grad_dfs[:, None, None] * hess_curve, axis=0)
            gammas = term1 + term2
            gammas = np.array(gammas, dtype=np.float64) * 1e-8
            out["gamma"] = Gamma(gammas, swap_times)

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
    

    @partial(jit, static_argnums=(0, 3, 4))  # self, disc_interp_type, idx_interp_type
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

        df_val   = disc_interp.simple_interpolate(value_time, times, dfs, disc_interp_type.value)
        df_start = idx_interp.simple_interpolate(start_times, times, dfs, idx_interp_type.value)
        df_end   = idx_interp.simple_interpolate(end_times, times, dfs, idx_interp_type.value)

        # d) Vectorised forward rates
        fwd = (df_start / df_end - 1.0) / pay_alphas                 # [..., M]

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

        df_pmts    = disc_interp.simple_interpolate(payment_times, times, dfs, disc_interp_type.value)
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
            out["value"] = Valuation(amount=float(val), currency=floating_leg_details._currency)

        need_grad = RequestTypes.DELTA in requests or RequestTypes.GAMMA in requests
        compute_hessian = RequestTypes.GAMMA in requests
        
        if need_grad:
            # Create a unique signature for this floating leg
            leg_signature = ("float", 
                           hash(tuple(payment_times.tolist() + start_times.tolist() + end_times.tolist())), 
                           hash(tuple(pay_alphas.tolist() + spreads.tolist() + notionals.tolist())),
                           principal, leg_sign, value_time, fix0, override_first)
            
            grad_dfs, hess_dfs = self._get_cached_gradients(
                cache, pv_fn, dfs, leg_signature, compute_hessian
            )

        if RequestTypes.DELTA in requests:
            sensitivities = jnp.dot(grad_dfs, jac)
            sensies = [float(x) * 1e-4 for x in sensitivities]
            out["delta"] = Delta(
                risk_ladder=sensies,
                tenors=to_tenor(swap_times),
                currency=floating_leg_details._currency,
                curve_type=CurveTypes.GBP_OIS_SONIA,
            )

        if RequestTypes.GAMMA in requests:
            term1 = jac.T @ hess_dfs @ jac
            term2 = jnp.sum(grad_dfs[:, None, None] * hess_curve, axis=0)
            gammas = term1 + term2
            gammas = np.array(gammas, dtype=np.float64) * 1e-8
            out["gamma"] = Gamma(gammas, swap_times)

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