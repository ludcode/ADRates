"""
Position class for valuing derivatives and computing risk sensitivities.

A Position wraps a derivative instrument and a market model, providing
a unified interface for computing PV, delta, gamma, and other analytics
via automatic differentiation through the Engine class.
"""

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
    """
    Represents a derivative position within a market model.

    Combines a derivative instrument with a market model and provides
    methods to compute valuations and sensitivities via the Engine.

    Attributes:
        derivative: The derivative instrument (e.g., OIS swap)
        model: Market model containing curves and conventions
        _engine: Internal Engine instance for computations

    Example:
        >>> swap = OIS(value_dt, "10Y", 0.04)
        >>> pos = swap.position(model)
        >>> result = pos.compute([RequestTypes.VALUE, RequestTypes.DELTA])
    """
    def __init__(self,
                 derivative,
                 model):
        """
        Create a position from a derivative and model.

        Args:
            derivative: Derivative instrument to value
            model: Market model with curves and conventions
        """
        self.derivative = derivative
        self.model = model

        self._engine = Engine(model)


        #TODO: Remove from lower classes and move to position()
        # self.amount = amount
        # self.direction = direction

    def compute(self, request_list, collateral_type=None):
        """
        Compute requested analytics for the position.

        Args:
            request_list (Iterable[RequestTypes]): List of analytics to compute
                (e.g., [RequestTypes.VALUE, RequestTypes.DELTA])
            collateral_type (CollateralType, optional): Type of collateral for discounting.
                If None, uses natural currency (default). For cross-currency collateral,
                specify the collateral currency (e.g., CollateralType.USD for USD collateral).

        Returns:
            AnalyticsResult: Object containing requested analytics (value, risk, gamma)

        Note:
            Delegates computation to the internal Engine which uses automatic
            differentiation for sensitivities.
        """
        return self._engine.compute(self.derivative, request_list, collateral_type)

