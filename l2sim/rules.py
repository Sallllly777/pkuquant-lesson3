from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from .data import parse_time

OPEN_START = parse_time("09:15:00")
OPEN_LOCK = parse_time("09:20:00")
OPEN_END = parse_time("09:25:00")
AM_START = parse_time("09:30:00")
AM_END = parse_time("11:30:00")
PM_START = parse_time("13:00:00")
CLOSE_START = parse_time("14:57:00")
CLOSE_END = parse_time("15:00:00")
AFTER_START = parse_time("15:05:00")
DAY_END = parse_time("15:30:00")


def phase(time):
    if OPEN_START <= time < OPEN_END:
        return "OPEN_AUCTION"
    if AM_START <= time < AM_END or PM_START <= time < CLOSE_START:
        return "CONTINUOUS"
    if CLOSE_START <= time < CLOSE_END:
        return "CLOSE_AUCTION"
    if AFTER_START <= time < DAY_END:
        return "AFTER_HOURS"
    return "BREAK" if time < CLOSE_END else "CLOSED"


def can_cancel(time):
    return OPEN_START <= time < OPEN_LOCK or phase(time) == "CONTINUOUS"


def input_price(value):
    try:
        d = Decimal(str(value))
        if not d.is_finite() or d <= 0 or d * 100 != (d * 100).to_integral_value():
            raise ValueError("Price must be positive and on a RMB 0.01 tick")
        return int(d * 100)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("Invalid price") from exc


def daily_limits(preclose, star):
    pct = Decimal("0.20" if star else "0.10")
    low, high = (int((Decimal(preclose) * (1 + sign * pct)).quantize(Decimal(1), rounding=ROUND_HALF_UP)) for sign in (-1, 1))
    return max(1, min(preclose - 1, low)), max(preclose + 1, high)


def validate_order(symbol, side, price, quantity, preclose, book, last_price, sellable, continuous=True):
    if side not in ("BUY", "SELL"):
        raise ValueError("Side must be BUY or SELL")
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("Quantity must be a positive integer")
    star = str(symbol).startswith("688")
    if quantity > (100000 if star else 1000000):
        raise ValueError("Quantity exceeds the limit-order maximum")
    if star:
        if quantity < 200 and not (side == "SELL" and quantity == sellable):
            raise ValueError("STAR limit orders require at least 200 shares (except selling an odd-lot balance)")
    elif side == "BUY" and quantity % 100:
        raise ValueError("Main-board purchases require multiples of 100 shares")
    elif side == "SELL" and quantity % 100 and quantity % 100 != sellable % 100:
        raise ValueError("The odd-lot remainder must be sold together, optionally with round lots")
    if not preclose:
        raise ValueError("Previous close has not been observed")
    low, high = daily_limits(preclose, star)
    if not low <= price <= high:
        raise ValueError(f"Outside daily price limits [{low / 100:.2f}, {high / 100:.2f}]")
    if not continuous:
        return
    opposite = "SELL" if side == "BUY" else "BUY"
    base = book.best(opposite) or book.best(side) or last_price or preclose
    if side == "BUY":
        cap = max(base + 1, (base * 102 + 50) // 100)
        if not star:
            cap = max(cap, base + 10)
        if price > cap:
            raise ValueError("Above the continuous-auction buy price cage")
    else:
        floor = max(1, min(base - 1, (base * 98 + 50) // 100))
        if not star:
            floor = max(1, min(floor, base - 10))
        if price < floor:
            raise ValueError("Below the continuous-auction sell price cage")
