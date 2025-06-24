"Create Position to value derivatives"

import jax
from functools import partial

from cavour.market.curves.interpolator import *
from cavour.utils.helpers import to_tenor, times_from_dates
from cavour.utils.date import Date
from cavour.market.curves.interpolator_ad import InterpolatorAd
from cavour.requests.results import Valuation, Risk, Delta, AnalyticsResult
from cavour.utils.global_types import (SwapTypes, 
                                   InstrumentTypes, 
                                   RequestTypes,
                                   CurveTypes)
from cavour.utils.currency import CurrencyTypes
from cavour.market.position.engine import Engine


class Position:
    def __init__(self,
                 derivative,
                 model):
        
        self.derivative = derivative
        self.model = model

        self._engine = Engine(model)


        #TODO: Remove from lower classes and move to position()
        # self.amount = amount
        # self.direction = direction

    def compute(self, request_list):

        compute_output = {}

        if self.derivative.derivative_type == InstrumentTypes.OIS_SWAP:
        
            for req in request_list:
                if req == RequestTypes.VALUE:
                    
                    compute_output['value'] = self._engine.valuation(self.derivative)

                if req == RequestTypes.DELTA:

                    compute_output['delta'] = self._engine.delta(self.derivative)

                if req == RequestTypes.GAMMA:

                    compute_output['gamma'] = self._engine.gamma(self.derivative)

        else:
            raise LibError(f"{self.derivative.derivative_type} not yet implemented")

        analytics_results = AnalyticsResult(value=compute_output['value'],
                                            risk=compute_output['delta'],
                                            gamma = compute_output['gamma'])

        return analytics_results

    # def _value(self):

    #     if self.derivative.derivative_type == InstrumentTypes.SWAP_FIXED_LEG:
    #         return self._value_swap_fixed_leg()
        
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
        
    # def _price_fixed_leg_jax(self,
    #                         swap_rates,                  # [..., N]
    #                         payment_times,               # [M]
    #                         payments,                    # [M]
    #                         principal: float,            # scalar
    #                         notional: float,             # scalar
    #                         leg_sign: float,             # +1 or −1
    #                         value_time: float,           # scalar
    #                         discount_curve_interpolator  # static object
    #                         ):
    #     # reconstruct the curve DFs at the knots:
    #     #times, dfs = self.build_curve_ad(swap_rates, swap_times, year_fracs)
    #     # discount at valuation date:    shape [...]
    #     df_val   = discount_curve_interpolator.interpolate(value_time)
    #     # discount at each payment date: shape [..., M]
    #     df_pmts  = discount_curve_interpolator.interpolate(payment_times)

    #     # build a mask of “after valuation date” over your M flows
    #     mask     = payment_times > value_time   # [M]
    #     # broadcast mask over any batch‐dimensions of df_pmts
    #     mask     = jnp.broadcast_to(mask, df_pmts.shape)

    #     # relative discount factors: shape [..., M]
    #     df_rel   = df_pmts / df_val[..., None]

    #     # PV of coupons:
    #     pv_coupons   = jnp.where(mask, payments * df_rel, 0.0)   # [..., M]
    #     # PV of final principal on last cash‐flow
    #     final_mask   = mask[..., -1]                            # [...]
    #     final_df_rel = df_rel[..., -1]                          # [...]
    #     pv_prin      = jnp.where(final_mask,
    #                             principal * final_df_rel,
    #                             0.0)                         # [...]

    #     # sum them up:
    #     leg_pv = jnp.sum(pv_coupons, axis=-1) + pv_prin          # [...]
    #     return leg_sign * leg_pv
    
    # def value_fixed_leg(self,
    #                     swap_rates, 
    #                     swap_times, 
    #                     year_fracs,
    #                     fixed_leg_details,
    #                     value_dt: Date,
    #                     interpolator_dc_type):
        
    #     #swap_rates =  [x*1e-4  for x in swap_rates]

    #     # — build the curve —
    #     times, dfs = self.build_curve_ad(swap_rates, swap_times, year_fracs)
    #     interp = InterpolatorAd(interpolator_dc_type)
    #     interp.fit(times=times, dfs=dfs)

    #     # — extract all the “static” pieces from your custom class ONCE —
    #     #    (these are plain Python numbers or JAX arrays)
    #     dc_type    = fixed_leg_details._dc_type
    #     # numeric offsets of each payment from the valuation date:
    #     payment_times = jnp.array([
    #         times_from_dates(dt, value_dt, dc_type)
    #         for dt in fixed_leg_details._payment_dts
    #     ])                               # shape [M]
    #     payments      = jnp.array(fixed_leg_details._payments)  # shape [M]
    #     principal     = fixed_leg_details._principal            # scalar
    #     notional      = fixed_leg_details._notional             # scalar
    #     leg_sign      = (
    #         +1.0 if fixed_leg_details._leg_type == SwapTypes.RECEIVE
    #         else -1.0
    #     )
    #     # numeric “value time”
    #     value_time = times_from_dates(value_dt, value_dt, dc_type)

    #     # — now call a tiny pure-JAX routine —
    #     pure_fn = partial(
    #         self._price_fixed_leg_jax,
    #         payment_times=payment_times,
    #         payments=payments,
    #         principal=principal,
    #         notional=notional,
    #         leg_sign=leg_sign,
    #         value_time=value_time,
    #         discount_curve_interpolator=interp,
    #     )
    #     return pure_fn(swap_rates)   # we only differentiate w.r.t. swap_rates

    # def valuation_fixed_leg(self,
    #                     swap_rates, 
    #                     swap_times, 
    #                     year_fracs,
    #                     fixed_leg_details,
    #                     value_dt: Date,
    #                     interpolator_dc_type):
        
    #     val = self.value_fixed_leg(
    #                     swap_rates, 
    #                     swap_times, 
    #                     year_fracs,
    #                     fixed_leg_details,
    #                     value_dt,
    #                     interpolator_dc_type
    #     )

    #     valuation = Valuation(amount=val.item(),
    #                           currency=CurrencyTypes.NONE)
        
    #     return valuation

    # def delta_fixed_leg(self,
    #                 swap_rates, 
    #                 swap_times, 
    #                 year_fracs,
    #                 fixed_leg_details,
    #                 value_dt: Date,
    #                 interpolator_dc_type): # Gradient w.r.t. swap_rates
    
    #     grad_price = jax.grad(lambda sr: self.value_fixed_leg(
    #         sr,
    #         swap_times, 
    #         year_fracs,
    #         fixed_leg_details,
    #         value_dt,
    #         interpolator_dc_type))
        
    #     sensitivities = grad_price(swap_rates)

    #     sensies = [x * 1e-4 for x in sensitivities]

    #     tenors = to_tenor(swap_times)

    #     delta= Delta(risk_ladder=sensies, 
    #           tenors=tenors,
    #           currency=CurrencyTypes.GBP, 
    #           curve_type=CurveTypes.GBP_OIS_SONIA)
    
    #     return delta
    

    # def _float_leg_jax(self,
    #                 swap_rates,                   # [..., N]
    #                 swap_times,                   # [N]
    #                 year_fracs,                   # [N]
    #                 payment_times,                # [M]
    #                 start_times,                  # [M]
    #                 end_times,                    # [M]
    #                 pay_alphas,                   # [M]
    #                 spreads,                      # [M]
    #                 notionals,                    # [M]
    #                 principal: float,             # scalar
    #                 leg_sign: float,              # +1 or –1
    #                 value_time: float,            # scalar
    #                 first_fixing_rate: float,
    #                 override_first,     # scalar
    #                 discount_curve_interpolator,  # static
    #                 index_curve_interpolator      # static
    #                 ):
    #     # a) Rebuild curve DFs at the knots
    #     #times, dfs = build_curve_ad(swap_rates, swap_times, year_fracs)

    #     # b) DF @ valuation date
    #     df_val   = discount_curve_interpolator.interpolate(value_time)          # [...,]

    #     # c) DF @ accrual start/end for forward rates
    #     df_start = index_curve_interpolator.interpolate(start_times)            # [..., M]
    #     df_end   = index_curve_interpolator.interpolate(end_times)              # [..., M]

    #     # d) Vectorised forward rates
    #     fwd = (df_start / df_end - 1.0) / pay_alphas                 # [..., M]

    #     # only override if the user actually passed a first_fixing_rate
    #     first_mask     = jnp.arange(fwd.shape[-1]) == 0             # [M]
    #     # make it match the batch dims
    #     first_mask_b   = jnp.broadcast_to(first_mask, fwd.shape)    # [..., M]

    #     # broadcast the static Python bool as well
    #     override_mask  = first_mask_b & override_first              # [..., M]

    #     # apply override only where override_mask is True
    #     fwd = jnp.where(override_mask, first_fixing_rate, fwd)      # [..., M]

    #     # e) coupon amounts
    #     cf_amounts = (fwd + spreads) * pay_alphas * notionals                    # [..., M]

    #     # f) DF @ payment dates
    #     df_pmts    = discount_curve_interpolator.interpolate(payment_times)      # [..., M]
    #     df_rel     = df_pmts / df_val[..., None]                                 # [..., M]

    #     # g) mask out past payments
    #     valid      = payment_times >= value_time                                 # [M]
    #     valid      = jnp.broadcast_to(valid, cf_amounts.shape)                  # [..., M]

    #     # h) PV of coupons + principal
    #     pv_coupons = jnp.where(valid, cf_amounts * df_rel, 0.0)                  # [..., M]
    #     pv_prin    = jnp.where(valid[..., -1],
    #                         principal * df_rel[..., -1],
    #                         0.0)                                            # [...]

    #     # i) aggregate and apply sign
    #     leg_pv     = jnp.sum(pv_coupons, axis=-1) + pv_prin                      # [...]
    #     return leg_sign * leg_pv
            
    # def value_float_leg(self,
    #                 swap_rates,
    #                 swap_times,
    #                 year_fracs,
    #                 floating_leg_details,
    #                 value_dt,
    #                 discount_curve_type,
    #                 index_curve_type = None,
    #                 first_fixing_rate = None):
    #     """
    #     Compute the floating‐leg PV, building both discount and index
    #     InterpolatorAd() objects from their interp‐type strings.
    #     """
    #     # 1) Build the discount curve
    #     times, dfs = self.build_curve_ad(swap_rates, swap_times, year_fracs)
    #     disc_interp = InterpolatorAd(discount_curve_type)
    #     disc_interp.fit(times=times, dfs=dfs)

    #     # 2) Build (or default) the index curve
    #     if index_curve_type is None:
    #         idx_interp = disc_interp
    #     else:
    #         idx_interp = InterpolatorAd(index_curve_type)
    #         idx_interp.fit(times=times, dfs=dfs)

    #     # 3) Extract all “static” inputs from your custom class & value_dt
    #     dc_type      = floating_leg_details._dc_type
    #     # payment, start, end offsets from value_dt → [M]
    #     payment_times = jnp.array([
    #         times_from_dates(dt, value_dt, dc_type)
    #         for dt in floating_leg_details._payment_dts
    #     ])
    #     start_times   = jnp.array([
    #         times_from_dates(dt0, value_dt, dc_type)
    #         for dt0 in floating_leg_details._start_accrued_dts
    #     ])
    #     end_times     = jnp.array([
    #         times_from_dates(dt1, value_dt, dc_type)
    #         for dt1 in floating_leg_details._end_accrued_dts
    #     ])

    #     pay_alphas    = jnp.array(floating_leg_details._year_fracs)        # [M]
    #     spreads       = jnp.full_like(pay_alphas, floating_leg_details._spread)
    #     notionals     = jnp.array(
    #         floating_leg_details._notional_array
    #         or [floating_leg_details._notional] * len(pay_alphas)
    #     )                                                                   # [M]

    #     principal     = floating_leg_details._principal                     # scalar
    #     leg_sign      = (+1.0 
    #                     if floating_leg_details._leg_type == SwapTypes.RECEIVE
    #                     else -1.0)
    #     value_time    = times_from_dates(value_dt, value_dt, dc_type)      # scalar
    #     override_first = first_fixing_rate is not None
    #     fix0          = first_fixing_rate if override_first else 0.0
    #     #fix0          = first_fixing_rate or 0.0                           # scalar

    #     # 4) Bind into a pure‐JAX function
    #     pure_fn = partial(
    #         self._float_leg_jax,
    #         swap_times=swap_times,
    #         year_fracs=year_fracs,
    #         payment_times=payment_times,
    #         start_times=start_times,
    #         end_times=end_times,
    #         pay_alphas=pay_alphas,
    #         spreads=spreads,
    #         notionals=notionals,
    #         principal=principal,
    #         leg_sign=leg_sign,
    #         value_time=value_time,
    #         first_fixing_rate=fix0,
    #         override_first=override_first,
    #         discount_curve_interpolator=disc_interp,
    #         index_curve_interpolator=idx_interp
    #     )

    #     # 5) Call it with swap_rates as the only differentiable arg
    #     return pure_fn(swap_rates)
    
    # def valuation_float_leg(self,
    #                 swap_rates,
    #                 swap_times,
    #                 year_fracs,
    #                 floating_leg_details,
    #                 value_dt,
    #                 discount_curve_type,
    #                 index_curve_type = None,
    #                 first_fixing_rate = None):

    #     val = self.value_float_leg(self,
    #                 swap_rates,
    #                 swap_times,
    #                 year_fracs,
    #                 floating_leg_details,
    #                 value_dt,
    #                 discount_curve_type,
    #                 index_curve_type,
    #                 first_fixing_rate)
        
    #     valuation = Valuation(amount=val.item(),
    #                           currency=CurrencyTypes.NONE)
        
    #     return valuation

    # def delta_float_leg(self,
    #                 swap_rates,
    #                 swap_times,
    #                 year_fracs,
    #                 floating_leg_details,
    #                 value_dt,
    #                 discount_curve_type,
    #                 index_curve_type = None,
    #                 first_fixing_rate = None):
        
    #     # Gradient w.r.t. swap_rates:
    #     delta = jax.grad(lambda sr: self.value_float_leg(
    #         sr, swap_times, year_fracs,
    #         floating_leg_details,
    #         value_dt,
    #         discount_curve_type,
    #         index_curve_type,
    #         first_fixing_rate
    #     ))

    #     sensitivities = delta(swap_rates)

    #     sensies = [x * 1e-4 for x in sensitivities]
    
    #     tenors = to_tenor(swap_times)

    #     delta= Delta(risk_ladder=sensies, 
    #           tenors=tenors,
    #           currency=CurrencyTypes.GBP, 
    #           curve_type=CurveTypes.GBP_OIS_SONIA)
    
    #     return delta


