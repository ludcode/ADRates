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
        return self._engine.compute(self.derivative, request_list)

