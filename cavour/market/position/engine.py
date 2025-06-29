
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


    # def build_curve_ad(self, swap_rates, swap_times, year_fracs):

    #     times = [] #jnp.array([])
    #     dfs = [] #jnp.array([])

    #     pv01 = 0.0
    #     df_settle = 1 

    #     pv01_dict = {}
    #     pv01 = 0

    #     def calculate_single_df(pv01, i, target_maturity=None, step=0):
    #         if target_maturity is None:
    #             t_mat = swap_times[i]
    #             #swap_rate = swap_rates[i]
    #         else:
    #             t_mat = target_maturity
    #             #swap_rate = interpolate_loglinear(t_mat)
    #         swap_rate = swap_rates[i]

    #         if len(year_fracs[i]) == 1:
    #             acc = year_fracs[i][0]
    #             pv01_end = (acc * swap_rate + 1.0)
    #             df_mat = (df_settle) / pv01_end
    #             pv01 = acc * df_mat
    #         else:
    #             acc = year_fracs[i][-1-step]
    #             last_payment = sum(year_fracs[i][:-1-step])
    #             if round(last_payment , 1) not in pv01_dict:
    #                 step += 1
    #                 pv01_dict[round(last_payment,1)] = calculate_single_df(pv01, i, last_payment, step)

    #             pv01_end = (acc * swap_rate + 1)
    #             df_mat = (df_settle - swap_rate * pv01_dict[round(last_payment,1)]) / pv01_end
    #             pv01 = pv01_dict[round(last_payment,1)] + acc * df_mat

    #         times.append(t_mat)
    #         dfs.append(df_mat)

    #         pv01_dict[round(t_mat,1)] = pv01

    #         step = 0

    #         return pv01
        
    #     for i in range(0, len(swap_rates)):
    #         pv01 = calculate_single_df(pv01, i)

    #     return times, dfs
    

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

        # 2) Precompute prev_idx & acc_last in Python (using 1-dp rounding)
        times_rounded = [round(t, 1) for t in swap_times]
        prev_idx = []
        acc_last = []
        for i, fr in enumerate(year_fracs):
            if len(fr) == 1:
                prev_idx.append(-1)
                acc_last.append(fr[0])
            else:
                last_pay = round(sum(fr[:-1]), 1)
                if last_pay in times_rounded:
                    j = times_rounded.index(last_pay)
                else:
                    # fallback: nearest tenor
                    diffs = [abs(t - last_pay) for t in swap_times]
                    j = int(min(range(len(diffs)), key=lambda k: diffs[k]))
                prev_idx.append(j)
                acc_last.append(fr[-1])

        prev_idx = jnp.array(prev_idx, dtype=jnp.int32)  # shape (N,)
        acc_last = jnp.array(acc_last)                   # shape (N,)

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

        build = lambda r: self.build_curve_ad(r, swap_times, year_fracs)[1]
        rates = jnp.array(swap_rates)
        times = jnp.array(swap_times)
        dfs = build(rates)

        jac = jacrev(build)(rates)
        hess = hessian(build)(rates)
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

    def valuation_fixed_leg(self,
                        swap_rates, 
                        swap_times, 
                        year_fracs,
                        fixed_leg_details,
                        value_dt: Date,
                        interpolator_dc_type):
        
        val = self.value_fixed_leg(
                        swap_rates, 
                        swap_times, 
                        year_fracs,
                        fixed_leg_details,
                        value_dt,
                        interpolator_dc_type
        )

        valuation = Valuation(amount=val.item(),
                              currency=fixed_leg_details._currency)
        
        return valuation

    def delta_fixed_leg(self,
                    swap_rates,
                    swap_times,
                    year_fracs,
                    fixed_leg_details,
                    value_dt: Date,
                    interpolator_dc_type): # Gradient w.r.t. swap_rates

        curve_key = tuple(swap_times)
        cache = self._cached_curve(curve_key, swap_rates, swap_times, year_fracs, interpolator_dc_type)
        times = cache["times"]
        dfs = cache["dfs"]
        jac = cache["jac"]

        dc_type    = fixed_leg_details._dc_type
        payment_times = jnp.array([
            times_from_dates(dt, value_dt, dc_type) for dt in fixed_leg_details._payment_dts
        ])
        payments      = jnp.array(fixed_leg_details._payments)
        principal     = fixed_leg_details._principal
        notional      = fixed_leg_details._notional
        leg_sign      = +1.0 if fixed_leg_details._leg_type == SwapTypes.RECEIVE else -1.0
        value_time    = times_from_dates(value_dt, value_dt, dc_type)

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

        grad_dfs = grad(lambda d: pv_fn(d))(dfs)
        sensitivities = jnp.dot(grad_dfs, jac)

        sensies = [float(x) * 1e-4 for x in sensitivities]

        tenors = to_tenor(swap_times)

        delta= Delta(risk_ladder=sensies, 
              tenors=tenors,
              currency=CurrencyTypes.GBP, 
              curve_type=CurveTypes.GBP_OIS_SONIA)
    
        return delta
    
    def gamma_fixed_leg(self,
                        swap_rates,
                        swap_times,
                        year_fracs,
                        fixed_leg_details,
                        value_dt: Date,
                        interpolator_dc_type):  # Hessian w.r.t. swap_rates
        curve_key = tuple(swap_times)
        cache = self._cached_curve(curve_key, swap_rates, swap_times, year_fracs, interpolator_dc_type)
        times = cache["times"]
        dfs = cache["dfs"]
        jac = cache["jac"]
        hess_curve = cache["hess"]

        dc_type    = fixed_leg_details._dc_type
        payment_times = jnp.array([times_from_dates(dt, value_dt, dc_type) for dt in fixed_leg_details._payment_dts])
        payments      = jnp.array(fixed_leg_details._payments)
        principal     = fixed_leg_details._principal
        notional      = fixed_leg_details._notional
        leg_sign      = +1.0 if fixed_leg_details._leg_type == SwapTypes.RECEIVE else -1.0
        value_time    = times_from_dates(value_dt, value_dt, dc_type)

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

        grad_dfs = grad(lambda d: pv_fn(d))(dfs)
        hess_dfs = hessian(lambda d: pv_fn(d))(dfs)

        term1 = jac.T @ hess_dfs @ jac
        term2 = jnp.sum(grad_dfs[:, None, None] * hess_curve, axis=0)
        gammas = term1 + term2

        gammas = np.array(gammas, dtype=np.float64) * 1e-8
        gamma = Gamma(gammas, swap_times)
        return gamma
    

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
    
    def valuation_float_leg(self,
                    swap_rates,
                    swap_times,
                    year_fracs,
                    floating_leg_details,
                    value_dt,
                    discount_curve_type,
                    index_curve_type = None,
                    first_fixing_rate = None):

        val = self.value_float_leg(
                    swap_rates,
                    swap_times,
                    year_fracs,
                    floating_leg_details,
                    value_dt,
                    discount_curve_type,
                    index_curve_type,
                    first_fixing_rate)
        
        valuation = Valuation(amount=val.item(),
                              currency=floating_leg_details._currency)
        
        return valuation

    def delta_float_leg(self,
                    swap_rates,
                    swap_times,
                    year_fracs,
                    floating_leg_details,
                    value_dt,
                    discount_curve_type,
                    index_curve_type = None,
                    first_fixing_rate = None):

        curve_key = tuple(swap_times)
        cache = self._cached_curve(curve_key, swap_rates, swap_times, year_fracs, discount_curve_type)
        times = cache["times"]
        dfs = cache["dfs"]
        jac = cache["jac"]

        dc_type      = floating_leg_details._dc_type
        payment_times = jnp.array([times_from_dates(dt, value_dt, dc_type) for dt in floating_leg_details._payment_dts])
        start_times   = jnp.array([times_from_dates(dt0, value_dt, dc_type) for dt0 in floating_leg_details._start_accrued_dts])
        end_times     = jnp.array([times_from_dates(dt1, value_dt, dc_type) for dt1 in floating_leg_details._end_accrued_dts])
        pay_alphas    = jnp.array(floating_leg_details._year_fracs)
        spreads       = jnp.full_like(pay_alphas, floating_leg_details._spread)
        notionals     = jnp.array(
            floating_leg_details._notional_array or [floating_leg_details._notional] * len(pay_alphas)
        )
        principal     = floating_leg_details._principal
        leg_sign      = +1.0 if floating_leg_details._leg_type == SwapTypes.RECEIVE else -1.0
        value_time    = times_from_dates(value_dt, value_dt, dc_type)
        override_first = first_fixing_rate is not None
        fix0          = first_fixing_rate if override_first else 0.0

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

        grad_dfs = grad(lambda d: pv_fn(d))(dfs)
        sensitivities = jnp.dot(grad_dfs, jac)

        sensies = [float(x) * 1e-4 for x in sensitivities]
    
        tenors = to_tenor(swap_times)

        delta= Delta(risk_ladder=sensies, 
              tenors=tenors,
              currency=CurrencyTypes.GBP, 
              curve_type=CurveTypes.GBP_OIS_SONIA)
    
        return delta
    
    def gamma_float_leg(self,
                    swap_rates,
                    swap_times,
                    year_fracs,
                    floating_leg_details,
                    value_dt,
                    discount_curve_type,
                    index_curve_type = None,
                    first_fixing_rate = None):
        curve_key = tuple(swap_times)
        cache = self._cached_curve(curve_key, swap_rates, swap_times, year_fracs, discount_curve_type)
        times = cache["times"]
        dfs = cache["dfs"]
        jac = cache["jac"]
        hess_curve = cache["hess"]

        dc_type      = floating_leg_details._dc_type
        payment_times = jnp.array([times_from_dates(dt, value_dt, dc_type) for dt in floating_leg_details._payment_dts])
        start_times   = jnp.array([times_from_dates(dt0, value_dt, dc_type) for dt0 in floating_leg_details._start_accrued_dts])
        end_times     = jnp.array([times_from_dates(dt1, value_dt, dc_type) for dt1 in floating_leg_details._end_accrued_dts])
        pay_alphas    = jnp.array(floating_leg_details._year_fracs)
        spreads       = jnp.full_like(pay_alphas, floating_leg_details._spread)
        notionals     = jnp.array(
            floating_leg_details._notional_array or [floating_leg_details._notional] * len(pay_alphas)
        )
        principal     = floating_leg_details._principal
        leg_sign      = +1.0 if floating_leg_details._leg_type == SwapTypes.RECEIVE else -1.0
        value_time    = times_from_dates(value_dt, value_dt, dc_type)
        override_first = first_fixing_rate is not None
        fix0          = first_fixing_rate if override_first else 0.0

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

        grad_dfs = grad(lambda d: pv_fn(d))(dfs)
        hess_dfs = hessian(lambda d: pv_fn(d))(dfs)

        term1 = jac.T @ hess_dfs @ jac
        term2 = jnp.sum(grad_dfs[:, None, None] * hess_curve, axis=0)
        gammas = term1 + term2

        gammas = np.array(gammas, dtype=np.float64) * 1e-8
        gamma = Gamma(gammas, swap_times)
        return gamma


