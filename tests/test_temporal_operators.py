from __future__ import annotations

import observable_library as ol


def test_delta_returns_latest_minus_lagged_value() -> None:
    assert ol.delta([1.0, 4.0, 9.0], lag=2) == 8.0


def test_ema_returns_exponential_moving_average() -> None:
    assert ol.ema([2.0, 4.0, 8.0], alpha=0.5) == 5.5


def test_slope_returns_linear_trend_over_indices() -> None:
    assert ol.slope([1.0, 3.0, 5.0]) == 2.0


def test_rolling_std_returns_population_std_for_latest_window() -> None:
    assert ol.rolling_std([1.0, 2.0, 3.0, 4.0], window=2) == 0.5
