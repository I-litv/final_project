from services.formatting import format_price_range, format_usd


def build_statistics_table(listings, group_columns, limit=None):
    grouped = (
        listings.groupby(group_columns, dropna=False)
        .agg(
            Listings=("price_usd", "size"),
            Min_Price=("price_usd", "min"),
            Max_Price=("price_usd", "max"),
            Median_Price=("price_usd", "median"),
        )
        .reset_index()
        .sort_values(["Listings", "Median_Price"], ascending=[False, False])
    )

    if limit:
        grouped = grouped.head(limit)

    grouped["Price Range"] = grouped.apply(
        lambda row: format_price_range(row["Min_Price"], row["Max_Price"]),
        axis=1,
    )
    grouped["Median Price"] = grouped["Median_Price"].apply(format_usd)
    grouped = grouped.drop(columns=["Min_Price", "Max_Price", "Median_Price"])

    return grouped
