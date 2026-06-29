def format_usd(value, decimals=0):
    """Format a number as a compact USD value."""
    return f"USD {float(value):,.{decimals}f}"


def format_price_range(min_price, max_price):
    return f"{format_usd(min_price)} - {format_usd(max_price)}"
