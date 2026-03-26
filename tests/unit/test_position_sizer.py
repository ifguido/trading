"""Comprehensive unit tests for PositionSizer.

Covers: fixed_fraction basic calc, zero equity, zero stop_distance,
max_position_pct cap, min_order_size filter, volatility_adjusted mode,
kelly mode, and kelly fallback.
"""

from __future__ import annotations

import pytest
from decimal import Decimal

from src.risk.position_sizer import PositionSizer, SizingMode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sizer() -> PositionSizer:
    """Default fixed-fraction sizer with 10% max position, tiny min order."""
    return PositionSizer(
        mode=SizingMode.FIXED_FRACTION,
        max_position_pct=Decimal("0.10"),
        min_order_size=Decimal("0.00001"),
    )


@pytest.fixture
def volatility_sizer() -> PositionSizer:
    return PositionSizer(
        mode=SizingMode.VOLATILITY_ADJUSTED,
        max_position_pct=Decimal("0.10"),
        min_order_size=Decimal("0.00001"),
    )


@pytest.fixture
def kelly_sizer() -> PositionSizer:
    return PositionSizer(
        mode=SizingMode.KELLY,
        max_position_pct=Decimal("0.10"),
        min_order_size=Decimal("0.00001"),
    )


# ---------------------------------------------------------------------------
# 1. fixed_fraction basic calculation
# ---------------------------------------------------------------------------

class TestFixedFractionBasic:
    def test_basic_calculation(self, sizer: PositionSizer) -> None:
        """qty = (equity * risk_per_trade) / stop_distance
        = (10000 * 0.02) / 100 = 200 / 100 = 2.0
        cap = (10000 * 0.10) / 100 = 10 -> 2.0 is below cap, not capped.
        """
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
        )
        assert qty == Decimal("2.00000000")

    def test_different_equity_levels(self, sizer: PositionSizer) -> None:
        qty_small = sizer.calculate(
            equity=Decimal("1000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
        )
        qty_large = sizer.calculate(
            equity=Decimal("100000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
        )
        # small: (1000*0.02)/100 = 0.2, cap = 1000*0.10/100 = 1.0 -> not capped
        assert qty_small == Decimal("0.20000000")
        # large: (100000*0.02)/100 = 20, cap = 100000*0.10/100 = 100 -> not capped
        assert qty_large == Decimal("20.00000000")

    def test_various_risk_per_trade(self, sizer: PositionSizer) -> None:
        # Lower risk -> smaller position
        qty_low = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.01"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
        )
        qty_high = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.05"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
        )
        # low: (10000*0.01)/100 = 1.0, cap = 10000*0.10/100 = 10 -> not capped
        assert qty_low == Decimal("1.00000000")
        # high: (10000*0.05)/100 = 5.0, cap = 10000*0.10/100 = 10 -> not capped
        assert qty_high == Decimal("5.00000000")


# ---------------------------------------------------------------------------
# 2. Returns 0 for zero equity
# ---------------------------------------------------------------------------

class TestZeroEquity:
    def test_zero_equity_returns_zero(self, sizer: PositionSizer) -> None:
        qty = sizer.calculate(
            equity=Decimal("0"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("50000"),
            stop_distance=Decimal("100"),
        )
        assert qty == Decimal("0")

    def test_negative_equity_returns_zero(self, sizer: PositionSizer) -> None:
        qty = sizer.calculate(
            equity=Decimal("-1000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("50000"),
            stop_distance=Decimal("100"),
        )
        assert qty == Decimal("0")

    def test_zero_entry_price_returns_zero(self, sizer: PositionSizer) -> None:
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("0"),
            stop_distance=Decimal("100"),
        )
        assert qty == Decimal("0")

    def test_negative_entry_price_returns_zero(self, sizer: PositionSizer) -> None:
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("-100"),
            stop_distance=Decimal("100"),
        )
        assert qty == Decimal("0")


# ---------------------------------------------------------------------------
# 3. Returns 0 for zero stop_distance
# ---------------------------------------------------------------------------

class TestZeroStopDistance:
    def test_zero_stop_distance_returns_zero(self, sizer: PositionSizer) -> None:
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("50000"),
            stop_distance=Decimal("0"),
        )
        assert qty == Decimal("0")

    def test_negative_stop_distance_returns_zero(self, sizer: PositionSizer) -> None:
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("50000"),
            stop_distance=Decimal("-50"),
        )
        assert qty == Decimal("0")


# ---------------------------------------------------------------------------
# 4. Respects max_position_pct cap
# ---------------------------------------------------------------------------

class TestMaxPositionPctCap:
    def test_cap_limits_large_position(self) -> None:
        """With a small cap, position should be limited."""
        sizer = PositionSizer(
            mode=SizingMode.FIXED_FRACTION,
            max_position_pct=Decimal("0.05"),  # 5% cap
            min_order_size=Decimal("0.00001"),
        )
        # Without cap: (10000 * 0.10) / 10 = 100
        # With cap: (10000 * 0.05) / 100 = 5
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.10"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("10"),
        )
        assert qty == Decimal("5.00000000")

    def test_no_cap_when_below_limit(self) -> None:
        """Position below cap should not be affected."""
        sizer = PositionSizer(
            mode=SizingMode.FIXED_FRACTION,
            max_position_pct=Decimal("0.50"),  # Very generous cap
            min_order_size=Decimal("0.00001"),
        )
        # qty = (10000 * 0.02) / 100 = 2
        # cap = (10000 * 0.50) / 50000 = 0.1
        # 2 > 0.1 so cap applies. Let's use a cheaper entry price.
        # cap = (10000 * 0.50) / 100 = 50
        # 2 < 50, so no cap
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
        )
        # qty = (10000 * 0.02) / 100 = 2.0
        # cap = 10000 * 0.50 / 100 = 50
        # 2 < 50, not capped
        assert qty == Decimal("2.00000000")

    def test_cap_with_different_entry_prices(self) -> None:
        sizer = PositionSizer(
            mode=SizingMode.FIXED_FRACTION,
            max_position_pct=Decimal("0.10"),
            min_order_size=Decimal("0.00001"),
        )
        # Expensive entry means cap bites sooner
        # qty = (10000 * 0.02) / 50 = 4
        # cap = (10000 * 0.10) / 500 = 2
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("500"),
            stop_distance=Decimal("50"),
        )
        assert qty == Decimal("2.00000000")


# ---------------------------------------------------------------------------
# 5. Returns 0 when below min_order_size
# ---------------------------------------------------------------------------

class TestMinOrderSize:
    def test_below_min_returns_zero(self) -> None:
        sizer = PositionSizer(
            mode=SizingMode.FIXED_FRACTION,
            max_position_pct=Decimal("0.10"),
            min_order_size=Decimal("1.0"),  # Relatively large minimum
        )
        # qty = (100 * 0.01) / 100 = 0.01
        # cap = (100 * 0.10) / 50000 = 0.0002
        # min(0.01, 0.0002) = 0.0002 < 1.0 -> return 0
        qty = sizer.calculate(
            equity=Decimal("100"),
            risk_per_trade=Decimal("0.01"),
            entry_price=Decimal("50000"),
            stop_distance=Decimal("100"),
        )
        assert qty == Decimal("0")

    def test_exactly_at_min_is_accepted(self) -> None:
        sizer = PositionSizer(
            mode=SizingMode.FIXED_FRACTION,
            max_position_pct=Decimal("1.0"),  # No cap effectively
            min_order_size=Decimal("1.0"),
        )
        # qty = (1000 * 0.1) / 100 = 1.0
        qty = sizer.calculate(
            equity=Decimal("1000"),
            risk_per_trade=Decimal("0.1"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
        )
        assert qty == Decimal("1.00000000")

    def test_above_min_is_accepted(self) -> None:
        sizer = PositionSizer(
            mode=SizingMode.FIXED_FRACTION,
            max_position_pct=Decimal("1.0"),
            min_order_size=Decimal("0.5"),
        )
        # qty = (10000 * 0.02) / 100 = 2.0
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
        )
        assert qty == Decimal("2.00000000")


# ---------------------------------------------------------------------------
# 6. volatility_adjusted mode
# ---------------------------------------------------------------------------

class TestVolatilityAdjusted:
    def test_high_volatility_reduces_size(self, volatility_sizer: PositionSizer) -> None:
        """High vol (4%) vs base vol (2%) should halve the position."""
        # qty = (10000 * 0.02) / (100 * (0.04 / 0.02))
        # = 200 / (100 * 2) = 200 / 200 = 1.0
        # cap = (10000 * 0.10) / 100 = 10 -> not capped
        qty = volatility_sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            volatility=Decimal("0.04"),
        )
        assert qty == Decimal("1.00000000")

    def test_low_volatility_increases_size(self, volatility_sizer: PositionSizer) -> None:
        """Low vol (1%) vs base vol (2%) -> multiplier = 0.5, so position doubles."""
        # vol_multiplier = 0.01 / 0.02 = 0.5, clamped to 0.5 (already 0.5)
        # qty = (10000 * 0.02) / (100 * 0.5) = 200 / 50 = 4.0
        # cap = 10000 * 0.10 / 100 = 10 -> not capped
        qty = volatility_sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            volatility=Decimal("0.01"),
        )
        assert qty == Decimal("4.00000000")

    def test_baseline_volatility_equals_fixed_fraction(self, volatility_sizer: PositionSizer) -> None:
        """At baseline vol (2%), result should equal fixed_fraction."""
        # vol_multiplier = 0.02 / 0.02 = 1.0
        # qty = (10000 * 0.02) / (100 * 1.0) = 2.0
        qty_vol = volatility_sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            volatility=Decimal("0.02"),
        )
        sizer_ff = PositionSizer(
            mode=SizingMode.FIXED_FRACTION,
            max_position_pct=Decimal("0.10"),
            min_order_size=Decimal("0.00001"),
        )
        qty_ff = sizer_ff.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
        )
        assert qty_vol == qty_ff

    def test_none_volatility_falls_back_to_fixed_fraction(self, volatility_sizer: PositionSizer) -> None:
        """No volatility provided -> fallback to fixed_fraction."""
        qty = volatility_sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            volatility=None,
        )
        # fixed_fraction: (10000 * 0.02) / 100 = 2.0
        # cap = 10000 * 0.10 / 100 = 10 -> not capped
        assert qty == Decimal("2.00000000")

    def test_zero_volatility_falls_back(self, volatility_sizer: PositionSizer) -> None:
        qty = volatility_sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            volatility=Decimal("0"),
        )
        # Fallback to fixed_fraction: (10000*0.02)/100 = 2.0
        # cap = 10000*0.10/100 = 10 -> not capped
        assert qty == Decimal("2.00000000")

    def test_very_high_volatility_capped_at_3x(self, volatility_sizer: PositionSizer) -> None:
        """Volatility multiplier is capped at 3.0."""
        # vol = 0.10, base_vol = 0.02, multiplier = 5.0 -> capped at 3.0
        # qty = (10000 * 0.02) / (100 * 3.0) = 200 / 300 = 0.666...
        # cap = 10000 * 0.10 / 100 = 10 -> not capped by position pct
        qty = volatility_sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            volatility=Decimal("0.10"),
        )
        # The code uses ROUND_DOWN, so 0.6666... truncates to 0.66666666
        assert qty == Decimal("0.66666666")


# ---------------------------------------------------------------------------
# 7. kelly mode
# ---------------------------------------------------------------------------

class TestKellyMode:
    def test_basic_kelly_calculation(self, kelly_sizer: PositionSizer) -> None:
        """
        Full Kelly = win_rate - (1 - win_rate) / payoff_ratio
                   = 0.6 - 0.4 / 2.0 = 0.6 - 0.2 = 0.4
        Half Kelly = 0.2
        risk_amount = 10000 * 0.2 = 2000
        qty = 2000 / 100 = 20
        cap = 10000 * 0.10 / 100 = 10
        -> capped at 10
        """
        qty = kelly_sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.50"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            win_rate=Decimal("0.6"),
            payoff_ratio=Decimal("2.0"),
        )
        assert qty == Decimal("10.00000000")

    def test_kelly_uncapped(self) -> None:
        """Use generous cap to verify the Kelly math itself."""
        sizer = PositionSizer(
            mode=SizingMode.KELLY,
            max_position_pct=Decimal("1.0"),
            min_order_size=Decimal("0.00001"),
        )
        # Full Kelly = 0.6 - 0.4/2.0 = 0.4
        # Half Kelly = 0.2
        # risk_amount = 10000 * 0.2 = 2000
        # qty = 2000 / 100 = 20
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.50"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            win_rate=Decimal("0.6"),
            payoff_ratio=Decimal("2.0"),
        )
        assert qty == Decimal("20.00000000")

    def test_kelly_negative_edge_returns_zero(self) -> None:
        """If Kelly fraction is negative, half-Kelly clamped to 0 -> qty = 0."""
        sizer = PositionSizer(
            mode=SizingMode.KELLY,
            max_position_pct=Decimal("1.0"),
            min_order_size=Decimal("0.00001"),
        )
        # Full Kelly = 0.3 - 0.7 / 0.5 = 0.3 - 1.4 = -1.1
        # Half Kelly = -0.55, clamped to 0
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            win_rate=Decimal("0.3"),
            payoff_ratio=Decimal("0.5"),
        )
        assert qty == Decimal("0")

    def test_kelly_half_kelly_capped_to_risk_per_trade(self) -> None:
        """Half-Kelly should be clamped at risk_per_trade as maximum."""
        sizer = PositionSizer(
            mode=SizingMode.KELLY,
            max_position_pct=Decimal("1.0"),
            min_order_size=Decimal("0.00001"),
        )
        # Full Kelly = 0.9 - 0.1 / 2.0 = 0.9 - 0.05 = 0.85
        # Half Kelly = 0.425, but risk_per_trade = 0.02
        # clamped to 0.02
        # qty = (10000 * 0.02) / 100 = 2.0
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            win_rate=Decimal("0.9"),
            payoff_ratio=Decimal("2.0"),
        )
        assert qty == Decimal("2.00000000")

    def test_kelly_zero_payoff_ratio_returns_zero(self) -> None:
        sizer = PositionSizer(
            mode=SizingMode.KELLY,
            max_position_pct=Decimal("1.0"),
            min_order_size=Decimal("0.00001"),
        )
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            win_rate=Decimal("0.6"),
            payoff_ratio=Decimal("0"),
        )
        assert qty == Decimal("0")

    def test_kelly_negative_payoff_ratio_returns_zero(self) -> None:
        sizer = PositionSizer(
            mode=SizingMode.KELLY,
            max_position_pct=Decimal("1.0"),
            min_order_size=Decimal("0.00001"),
        )
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            win_rate=Decimal("0.6"),
            payoff_ratio=Decimal("-1"),
        )
        assert qty == Decimal("0")


# ---------------------------------------------------------------------------
# 8. kelly falls back to fixed_fraction without win_rate
# ---------------------------------------------------------------------------

class TestKellyFallback:
    def test_fallback_no_win_rate(self, kelly_sizer: PositionSizer) -> None:
        """Missing win_rate -> fall back to fixed_fraction."""
        qty_kelly_fallback = kelly_sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            win_rate=None,
            payoff_ratio=Decimal("2.0"),
        )
        sizer_ff = PositionSizer(
            mode=SizingMode.FIXED_FRACTION,
            max_position_pct=Decimal("0.10"),
            min_order_size=Decimal("0.00001"),
        )
        qty_ff = sizer_ff.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
        )
        assert qty_kelly_fallback == qty_ff

    def test_fallback_no_payoff_ratio(self, kelly_sizer: PositionSizer) -> None:
        """Missing payoff_ratio -> fall back to fixed_fraction."""
        qty_kelly_fallback = kelly_sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            win_rate=Decimal("0.6"),
            payoff_ratio=None,
        )
        sizer_ff = PositionSizer(
            mode=SizingMode.FIXED_FRACTION,
            max_position_pct=Decimal("0.10"),
            min_order_size=Decimal("0.00001"),
        )
        qty_ff = sizer_ff.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
        )
        assert qty_kelly_fallback == qty_ff

    def test_fallback_both_none(self, kelly_sizer: PositionSizer) -> None:
        """Both win_rate and payoff_ratio None -> fall back to fixed_fraction."""
        qty = kelly_sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
            win_rate=None,
            payoff_ratio=None,
        )
        # fixed_fraction: (10000 * 0.02) / 100 = 2.0
        # cap: 10000 * 0.10 / 100 = 10 -> not capped
        assert qty == Decimal("2.00000000")


# ---------------------------------------------------------------------------
# Additional edge cases for SizingMode enum
# ---------------------------------------------------------------------------

class TestSizingModeEnum:
    def test_enum_values(self) -> None:
        assert SizingMode.FIXED_FRACTION.value == "fixed_fraction"
        assert SizingMode.VOLATILITY_ADJUSTED.value == "volatility_adjusted"
        assert SizingMode.KELLY.value == "kelly"

    def test_enum_from_string(self) -> None:
        assert SizingMode("fixed_fraction") == SizingMode.FIXED_FRACTION
        assert SizingMode("volatility_adjusted") == SizingMode.VOLATILITY_ADJUSTED
        assert SizingMode("kelly") == SizingMode.KELLY


# ---------------------------------------------------------------------------
# Rounding behaviour
# ---------------------------------------------------------------------------

class TestRounding:
    def test_result_is_rounded_down(self) -> None:
        """Quantities should always be rounded DOWN (ROUND_DOWN), not nearest."""
        sizer = PositionSizer(
            mode=SizingMode.FIXED_FRACTION,
            max_position_pct=Decimal("1.0"),
            min_order_size=Decimal("0.00001"),
        )
        # qty = (10000 * 0.02) / 3 = 66.666...
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("3"),
        )
        # Should be truncated, not rounded up
        assert qty == Decimal("66.66666666")

    def test_quantize_precision(self) -> None:
        """Result should have exactly 8 decimal places."""
        sizer = PositionSizer(
            mode=SizingMode.FIXED_FRACTION,
            max_position_pct=Decimal("1.0"),
            min_order_size=Decimal("0.00001"),
        )
        qty = sizer.calculate(
            equity=Decimal("10000"),
            risk_per_trade=Decimal("0.02"),
            entry_price=Decimal("100"),
            stop_distance=Decimal("100"),
        )
        # qty = 2.0 -> 2.00000000
        assert str(qty) == "2.00000000"
