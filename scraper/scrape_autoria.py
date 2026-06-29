import argparse
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_URL = "https://auto.ria.com"
SEARCH_URL = (
    "https://auto.ria.com/uk/search/"
    "?search_type=1&category=1&abroad=0&customs_cleared=1"
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "used_cars.csv"
MAX_LISTINGS = 50000
OUTPUT_COLUMNS = [
    "title",
    "make",
    "model",
    "year",
    "mileage_km",
    "price_usd",
    "fuel",
    "gearbox",
    "city",
    "listing_date",
    "url",
]
REQUIRED_MODEL_COLUMNS = ["make", "model", "year", "mileage_km", "price_usd"]
TEXT_COLUMNS = [
    "title",
    "make",
    "model",
    "fuel",
    "gearbox",
    "city",
    "listing_date",
    "url",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "uk-UA,uk;q=0.9,en;q=0.8",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/webp,*/*;q=0.8"
    ),
    "Connection": "keep-alive",
}


def clean_text(value):
    if value is None:
        return None
    if pd.isna(value):
        return None

    value = str(value).replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_for_key(value):
    cleaned_value = clean_text(value)
    return cleaned_value.lower() if cleaned_value else ""


def listing_key(record):
    """Build a duplicate key from the listing fields used by the model."""
    try:
        year = int(float(record.get("year")))
        mileage_km = int(float(record.get("mileage_km")))
        price_usd = int(float(record.get("price_usd")))
    except (TypeError, ValueError):
        return None

    make = normalize_for_key(record.get("make"))
    model = normalize_for_key(record.get("model"))
    if not make or not model:
        return None

    return (make, model, year, mileage_km, price_usd)


def extract_price_usd(text):
    if not text:
        return None

    match = re.search(r"([\d\s]+)\s*\$", text)
    if not match:
        return None

    price = match.group(1).replace(" ", "")
    try:
        return int(price)
    except ValueError:
        return None


def extract_year(text):
    if not text:
        return None

    match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if not match:
        return None

    return int(match.group(1))


def extract_mileage_km(text):
    if text is None:
        return None

    cleaned_text = clean_text(text)
    if cleaned_text is None:
        return None

    if "Без пробігу" in cleaned_text:
        return 0

    match = re.search(r"(\d+)\s*тис\.?\s*км", cleaned_text)
    if match:
        return int(match.group(1)) * 1000

    match = re.search(r"(\d+)\s*км", cleaned_text)
    if match:
        return int(match.group(1))

    return None


def extract_listing_date(text, today=None):
    """Convert AUTO.RIA relative listing text into an approximate ISO date."""
    if not text:
        return None

    today = today or datetime.now().date()
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return None

    lower_text = cleaned_text.lower()

    if "сьогодні" in lower_text:
        return today.isoformat()

    if "вчора" in lower_text or "день тому" in lower_text:
        return (today - timedelta(days=1)).isoformat()

    relative_patterns = [
        (r"(\d+)\s*(?:хв|хвилин|хвилини|хвилину)\s*тому", 0),
        (r"(\d+)\s*(?:год|годин|години|годину)\s*тому", 0),
        (r"(\d+)\s*(?:дн|дні|дня|днів)\s*тому", 1),
        (r"(\d+)\s*(?:тиж|тижнів|тижні|тижня)\s*тому", 7),
        (r"(\d+)\s*(?:міс|місяців|місяці|місяця)\s*тому", 30),
    ]

    for pattern, days_multiplier in relative_patterns:
        match = re.search(pattern, lower_text)
        if match:
            amount = int(match.group(1))
            days_ago = amount * days_multiplier
            return (today - timedelta(days=days_ago)).isoformat()

    date_match = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b", lower_text)
    if date_match:
        day, month, year = map(int, date_match.groups())
        try:
            return datetime(year, month, day).date().isoformat()
        except ValueError:
            return None

    return None


def extract_city(text):
    if not text:
        return None

    common_cities = [
        "Київ",
        "Львів",
        "Одеса",
        "Дніпро",
        "Харків",
        "Вінниця",
        "Запоріжжя",
        "Івано-Франківськ",
        "Хмельницький",
        "Чернівці",
        "Тернопіль",
        "Луцьк",
        "Рівне",
        "Житомир",
        "Полтава",
        "Черкаси",
        "Чернігів",
        "Суми",
        "Кропивницький",
        "Миколаїв",
        "Херсон",
        "Ужгород",
    ]

    for city in common_cities:
        if city in text:
            return city

    return None


def extract_fuel(text):
    if not text:
        return None

    fuel_options = [
        "Газ пропан-бутан / Бензин",
        "Газ метан / Бензин",
        "Бензин",
        "Дизель",
        "Електро",
        "Гібрид",
        "Газ",
    ]

    for fuel in fuel_options:
        if fuel in text:
            return fuel

    return None


def extract_gearbox(text):
    if not text:
        return None

    gearbox_options = [
        "Ручна / Механіка",
        "Автомат",
        "Робот",
        "Варіатор",
        "Типтронік",
        "Редуктор",
    ]

    for gearbox in gearbox_options:
        if gearbox in text:
            return gearbox

    return None


def split_make_model_title(title):
    if not title:
        return None, None

    title = clean_text(title)
    if not title:
        return None, None

    title = re.sub(r"^AUTO\.RIA\s*", "", title)
    title = re.sub(r"^Продаж\s+", "", title)
    title = re.sub(r"^Купити\s+", "", title)
    title = re.sub(r"^(Перевірений VIN|ТОП|Новий|В наявності)\s+", "", title)

    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", title)
    if year_match:
        title = title[: year_match.start()].strip()

    parts = title.split()
    if not parts:
        return None, None

    if len(parts) == 1:
        return parts[0], None

    return parts[0], parts[1]


def get_search_page(page):
    response = requests.get(
        SEARCH_URL,
        params={"page": page},
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def find_listing_cards(soup):
    selectors = [
        "a.product-card[data-car-id]",
        "[data-car-id]",
        "section.ticket-item",
        "div.ticket-item",
        "[data-advertisement-id]",
        ".content-bar",
    ]

    for selector in selectors:
        cards = soup.select(selector)
        if cards:
            return cards

    return []


def parse_listing_card(card, today=None):
    full_text = clean_text(card.get_text(" ", strip=True))
    feature_values = [
        value
        for value in (
            clean_text(feature.get_text(" ", strip=True))
            for feature in card.select(".grid-wrapper span.common-text")
        )
        if value
    ]

    if card.name == "a" and card.get("href"):
        link_tag = card
    else:
        link_tag = (
            card.select_one("a.product-card[href]")
            or card.select_one("a[href*='/auto_']")
            or card.select_one("a[href*='auto_']")
            or card.select_one("a[href]")
        )

    if not link_tag or not link_tag.get("href"):
        return None

    href = link_tag.get("href")
    if not isinstance(href, str):
        return None

    url = urljoin(BASE_URL, href)
    if "/auto_" not in url:
        return None

    title_tag = (
        card.select_one(".product-card-content .titleS")
        or card.select_one(".address")
        or card.select_one(".ticket-title")
        or card.select_one(".head-ticket")
        or card.select_one("a[href*='/auto_']")
        or link_tag
    )
    title = clean_text(title_tag.get_text(" ", strip=True)) if title_tag else None

    price_tag = (
        card.select_one(".product-card-content .titleM.c-green")
        or card.select_one(".product-card-content .c-green")
    )
    price_text = clean_text(price_tag.get_text(" ", strip=True)) if price_tag else None

    price_usd = extract_price_usd(price_text) or extract_price_usd(full_text)
    year = extract_year(title) or extract_year(full_text)
    mileage_km = extract_mileage_km(feature_values[0]) if feature_values else None

    if mileage_km is None:
        mileage_km = extract_mileage_km(full_text)

    fuel_text = next((value for value in feature_values if extract_fuel(value)), full_text)
    gearbox_text = next(
        (value for value in feature_values if extract_gearbox(value)),
        full_text,
    )

    fuel = extract_fuel(fuel_text)
    gearbox = extract_gearbox(gearbox_text)
    city = feature_values[-1] if feature_values else extract_city(full_text)
    make, model = split_make_model_title(title)

    if not url or not title:
        return None

    return {
        "title": title,
        "make": make,
        "model": model,
        "year": year,
        "mileage_km": mileage_km,
        "price_usd": price_usd,
        "fuel": fuel,
        "gearbox": gearbox,
        "city": city,
        "listing_date": extract_listing_date(full_text, today=today),
        "url": url,
    }


def parse_search_page(html):
    soup = BeautifulSoup(html, "html.parser")
    cards = find_listing_cards(soup)
    today = datetime.now().date()

    listings = []
    for card in cards:
        item = parse_listing_card(card, today=today)
        if item is None:
            continue
        if item["price_usd"] is None:
            continue
        if item["year"] is None:
            continue
        if item["make"] is None:
            continue
        if item["model"] is None:
            continue
        if item["mileage_km"] is None:
            continue

        listings.append(item)

    return listings


def empty_output_dataframe():
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def clean_listing_dataframe(df):
    """Return valid rows in the final CSV format used by the model."""
    if df.empty and not set(REQUIRED_MODEL_COLUMNS).issubset(df.columns):
        return empty_output_dataframe()

    missing_columns = [
        column for column in REQUIRED_MODEL_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        print(
            "Existing CSV is missing required columns and will be rebuilt: "
            + ", ".join(missing_columns)
        )
        return empty_output_dataframe()

    cleaned = df.copy()
    for column in OUTPUT_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = None

    for column in TEXT_COLUMNS:
        cleaned[column] = cleaned[column].apply(clean_text)

    cleaned["year"] = pd.to_numeric(cleaned["year"], errors="coerce")
    cleaned["mileage_km"] = pd.to_numeric(cleaned["mileage_km"], errors="coerce")
    cleaned["price_usd"] = pd.to_numeric(cleaned["price_usd"], errors="coerce")
    cleaned = cleaned.dropna(subset=REQUIRED_MODEL_COLUMNS)

    current_year = datetime.now().year
    cleaned = cleaned[
        (cleaned["make"].apply(normalize_for_key) != "")
        & (cleaned["model"].apply(normalize_for_key) != "")
        & (cleaned["year"] >= 1990)
        & (cleaned["year"] <= current_year)
        & (cleaned["mileage_km"] >= 0)
        & (cleaned["price_usd"] > 0)
    ].copy()

    if cleaned.empty:
        return empty_output_dataframe()

    cleaned["year"] = cleaned["year"].astype(int)
    cleaned["mileage_km"] = cleaned["mileage_km"].astype(int)
    cleaned["price_usd"] = cleaned["price_usd"].astype(int)

    cleaned["_listing_key"] = cleaned.apply(
        lambda row: listing_key(row.to_dict()),
        axis=1,
    )
    cleaned = cleaned.dropna(subset=["_listing_key"])

    if "url" in cleaned.columns:
        rows_with_url = cleaned["url"].notna() & (cleaned["url"] != "")
        cleaned_with_url = cleaned[rows_with_url].drop_duplicates(
            subset=["url"],
            keep="first",
        )
        cleaned_without_url = cleaned[~rows_with_url]
        cleaned = pd.concat(
            [cleaned_with_url, cleaned_without_url],
            ignore_index=True,
        )

    cleaned = cleaned.drop_duplicates(subset=["_listing_key"], keep="first")
    cleaned = cleaned.drop(columns=["_listing_key"])
    return cleaned[OUTPUT_COLUMNS]


def write_cleaned_data(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def get_seen_listing_keys(records):
    return {
        key
        for key in (listing_key(record) for record in records)
        if key is not None
    }


def get_records_and_seen_sets(df):
    records = df.to_dict("records")
    seen_urls = set(df["url"].dropna().tolist()) if "url" in df.columns else set()
    seen_listing_keys = get_seen_listing_keys(records)
    return records, seen_urls, seen_listing_keys


def load_existing_data(output_path):
    output_path = Path(output_path)
    if not output_path.exists():
        print("No existing CSV found. Starting with an empty cleaned dataset.")
        return [], set(), set()

    try:
        df = pd.read_csv(output_path)
        original_count = len(df)
        df = clean_listing_dataframe(df)
        cleaned_count = len(df)

        print(
            f"Cleaned existing CSV before scraping: "
            f"{original_count} -> {cleaned_count} rows."
        )
        write_cleaned_data(df, output_path)

        records, seen_urls, seen_listing_keys = get_records_and_seen_sets(df)

        print(f"Loaded existing file with {len(records)} records.")
        return records, seen_urls, seen_listing_keys
    except Exception as error:
        print(f"Could not clean existing CSV before scraping: {error}")
        return [], set(), set()


def save_data(records, output_path):
    df = pd.DataFrame(records)
    df = clean_listing_dataframe(df)
    write_cleaned_data(df, output_path)
    return df


def scrape_autoria(
    max_listings=MAX_LISTINGS,
    start_page=0,
    output_path=DEFAULT_OUTPUT_PATH,
    min_delay=1.5,
    max_delay=3.5,
):
    output_path = Path(output_path)
    all_listings, seen_urls, seen_listing_keys = load_existing_data(output_path)
    page = start_page

    progress = tqdm(
        total=max_listings,
        initial=min(len(all_listings), max_listings),
        desc="Scraping AUTO.RIA",
    )

    while len(all_listings) < max_listings:
        try:
            print(f"\nScraping page {page}...")

            html = get_search_page(page)
            listings = parse_search_page(html)

            print(f"Found {len(listings)} valid listings on page {page}.")
            if not listings:
                print(
                    "No listings found. The page structure may have changed "
                    "or scraping was blocked."
                )
                break

            saved_count_before_page = len(all_listings)
            new_items = 0
            for item in listings:
                if item["url"] in seen_urls:
                    continue

                key = listing_key(item)
                if key is None or key in seen_listing_keys:
                    continue

                seen_urls.add(item["url"])
                seen_listing_keys.add(key)
                all_listings.append(item)
                new_items += 1
                progress.update(1)

                if len(all_listings) >= max_listings:
                    break

            saved_df = save_data(all_listings, output_path)
            all_listings, seen_urls, seen_listing_keys = get_records_and_seen_sets(
                saved_df
            )
            cleaned_new_items = len(all_listings) - saved_count_before_page
            print(
                f"Added {cleaned_new_items} new listings after cleaning. "
                f"Total saved: {len(all_listings)}"
            )

            page += 1
            time.sleep(random.uniform(min_delay, max_delay))

        except requests.exceptions.HTTPError as error:
            print(f"HTTP error on page {page}: {error}")
            saved_df = save_data(all_listings, output_path)
            all_listings, seen_urls, seen_listing_keys = get_records_and_seen_sets(
                saved_df
            )
            time.sleep(10)
            page += 1

        except requests.exceptions.RequestException as error:
            print(f"Request error on page {page}: {error}")
            saved_df = save_data(all_listings, output_path)
            all_listings, seen_urls, seen_listing_keys = get_records_and_seen_sets(
                saved_df
            )
            time.sleep(10)
            page += 1

        except KeyboardInterrupt:
            print("\nScraping stopped by user.")
            saved_df = save_data(all_listings, output_path)
            all_listings, seen_urls, seen_listing_keys = get_records_and_seen_sets(
                saved_df
            )
            break

        except Exception as error:
            print(f"Unexpected error on page {page}: {error}")
            saved_df = save_data(all_listings, output_path)
            all_listings, seen_urls, seen_listing_keys = get_records_and_seen_sets(
                saved_df
            )
            time.sleep(10)
            page += 1

    progress.close()
    df = save_data(all_listings, output_path)

    print("\nDone.")
    print(f"Saved {len(df)} listings to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Scrape used car listings.")
    parser.add_argument(
        "--max-listings",
        type=int,
        default=MAX_LISTINGS,
        help="How many listings to collect.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=0,
        help="AUTO.RIA search page to start from.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to save the CSV output.",
    )
    parser.add_argument("--min-delay", type=float, default=1.5)
    parser.add_argument("--max-delay", type=float, default=3.5)
    args = parser.parse_args()

    scrape_autoria(
        max_listings=args.max_listings,
        start_page=args.start_page,
        output_path=args.output,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
    )


if __name__ == "__main__":
    main()
