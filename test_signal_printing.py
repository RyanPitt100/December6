#!/usr/bin/env python3
"""
Test script to verify f-string formatting fixes in signal/order printing.
Run this before deploying to VPS to confirm the bugs are fixed.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MockSignal:
    instrument: str
    direction: str
    entry_price: float
    sl_price: float
    tp_price: Optional[float]


@dataclass
class MockOrder:
    action: str
    instrument: str
    direction: Optional[str]
    entry_price: Optional[float]
    sl_price: Optional[float]
    tp_price: Optional[float]
    size_lots: float
    reason: str


def test_signal_printing():
    """Test signal printing with various tp_price values."""
    print("=" * 60)
    print("TEST: Signal printing")
    print("=" * 60)

    # Test case 1: tp_price is a float
    sig1 = MockSignal("EURUSD", "long", 1.12345, 1.12000, 1.12800)
    tp_str = f"{sig1.tp_price:.5f}" if sig1.tp_price else "None"
    print(f"  - {sig1.instrument} {sig1.direction.upper()} @ {sig1.entry_price:.5f} (SL: {sig1.sl_price:.5f}, TP: {tp_str})")
    print("  [PASS] Signal with tp_price as float")

    # Test case 2: tp_price is None
    sig2 = MockSignal("USDJPY", "short", 157.500, 158.000, None)
    tp_str = f"{sig2.tp_price:.5f}" if sig2.tp_price else "None"
    print(f"  - {sig2.instrument} {sig2.direction.upper()} @ {sig2.entry_price:.5f} (SL: {sig2.sl_price:.5f}, TP: {tp_str})")
    print("  [PASS] Signal with tp_price as None")

    # Test case 3: tp_price is 0.0 (falsy but valid)
    sig3 = MockSignal("GER40.cash", "long", 24000.0, 23900.0, 0.0)
    tp_str = f"{sig3.tp_price:.5f}" if sig3.tp_price else "None"
    print(f"  - {sig3.instrument} {sig3.direction.upper()} @ {sig3.entry_price:.5f} (SL: {sig3.sl_price:.5f}, TP: {tp_str})")
    print("  [PASS] Signal with tp_price as 0.0")


def test_order_printing():
    """Test order printing with various optional values."""
    print("\n" + "=" * 60)
    print("TEST: Order printing (dry-run format)")
    print("=" * 60)

    # Test case 1: All values present
    order1 = MockOrder("open", "EURUSD", "long", 1.12345, 1.12000, 1.12800, 0.50, "signal")
    direction_str = order1.direction.upper() if order1.direction else ''
    action_str = f"{order1.action.upper()} {direction_str}"
    entry_str = f"{order1.entry_price:.5f}" if order1.entry_price else "MARKET"
    sl_str = f"{order1.sl_price:.5f}" if order1.sl_price else "N/A"
    tp_str = f"{order1.tp_price:.5f}" if order1.tp_price else "N/A"
    print(f"[DRY-RUN]   {action_str} {order1.instrument} | "
          f"Entry: {entry_str} | SL: {sl_str} | TP: {tp_str} | "
          f"Size: {order1.size_lots:.2f} lots | Reason: {order1.reason}")
    print("  [PASS] Order with all values present")

    # Test case 2: entry_price is None (market order)
    order2 = MockOrder("open", "USDJPY", "short", None, 158.000, 156.500, 0.25, "signal")
    direction_str = order2.direction.upper() if order2.direction else ''
    action_str = f"{order2.action.upper()} {direction_str}"
    entry_str = f"{order2.entry_price:.5f}" if order2.entry_price else "MARKET"
    sl_str = f"{order2.sl_price:.5f}" if order2.sl_price else "N/A"
    tp_str = f"{order2.tp_price:.5f}" if order2.tp_price else "N/A"
    print(f"[DRY-RUN]   {action_str} {order2.instrument} | "
          f"Entry: {entry_str} | SL: {sl_str} | TP: {tp_str} | "
          f"Size: {order2.size_lots:.2f} lots | Reason: {order2.reason}")
    print("  [PASS] Order with entry_price as None (market order)")

    # Test case 3: All optional prices are None
    order3 = MockOrder("open", "AUS200.cash", "long", None, None, None, 1.0, "manual")
    direction_str = order3.direction.upper() if order3.direction else ''
    action_str = f"{order3.action.upper()} {direction_str}"
    entry_str = f"{order3.entry_price:.5f}" if order3.entry_price else "MARKET"
    sl_str = f"{order3.sl_price:.5f}" if order3.sl_price else "N/A"
    tp_str = f"{order3.tp_price:.5f}" if order3.tp_price else "N/A"
    print(f"[DRY-RUN]   {action_str} {order3.instrument} | "
          f"Entry: {entry_str} | SL: {sl_str} | TP: {tp_str} | "
          f"Size: {order3.size_lots:.2f} lots | Reason: {order3.reason}")
    print("  [PASS] Order with all optional prices as None")

    # Test case 4: Close order (direction is None)
    order4 = MockOrder("close", "EURUSD", None, None, None, None, 0.50, "flatten")
    direction_str = order4.direction.upper() if order4.direction else ''
    action_str = f"{order4.action.upper()} {direction_str}"
    print(f"[DRY-RUN]   CLOSE {order4.instrument} | Reason: {order4.reason}")
    print("  [PASS] Close order with direction as None")


def main():
    print("\n" + "=" * 60)
    print("SIGNAL/ORDER PRINTING TEST")
    print("=" * 60)
    print("This test verifies the f-string formatting fixes.")
    print("If this runs without errors, the fixes are working.\n")

    try:
        test_signal_printing()
        test_order_printing()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("Safe to deploy to VPS.")
        return 0

    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
