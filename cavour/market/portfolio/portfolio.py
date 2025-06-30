from typing import Iterable, List

from cavour.market.position.position import Position
from cavour.requests.results import AnalyticsResult
from cavour.utils.global_types import RequestTypes


class Portfolio:
    """Container aggregating multiple :class:`Position` objects."""

    def __init__(self, positions: Iterable[Position] | None = None) -> None:
        self._positions: List[Position] = list(positions or [])

    def add_position(self, position: Position) -> None:
        self._positions.append(position)

    #@property
    def positions(self) -> List[Position]:
        return list(self._positions)

    def compute(self, request_list: Iterable[RequestTypes]) -> AnalyticsResult:
        """Aggregate analytics for all positions in the portfolio."""
        total_val = None
        total_delta = None
        total_gamma = None

        for pos in self._positions:
            res = pos.compute(request_list)

            if RequestTypes.VALUE in request_list:
                if total_val is None:
                    total_val = res.value
                else:
                    total_val = total_val + res.value

            if RequestTypes.DELTA in request_list:
                if total_delta is None:
                    total_delta = res.risk
                else:
                    total_delta = total_delta + res.risk

            if RequestTypes.GAMMA in request_list:
                if total_gamma is None:
                    total_gamma = res.gamma
                else:
                    total_gamma = total_gamma + res.gamma

        return AnalyticsResult(value=total_val, risk=total_delta, gamma=total_gamma)

