import os
import re
import time
import random
import requests
import pandas as pd

from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin


BASE_URL = "https://auto.ria.com"
SEARCH_URL = "https://auto.ria.com/uk/search/"

OUTPUT_PATH = "data/autoria_listings.csv"
MAX_LISTINGS = 10_000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "uk-UA,uk;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Connection": "keep-alive",
}


def clean_text(value):
    if value is None:
        return None

    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


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


def extract_mileage_km(text: str | None) -> int | None:
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
        "Бензин",
        "Дизель",
        "Електро",
        "Гібрид",
        "Газ",
        "Газ пропан-бутан / Бензин",
        "Газ метан / Бензин",
    ]

    for fuel in fuel_options:
        if fuel in text:
            return fuel

    return None


def extract_gearbox(text):
    if not text:
        return None

    gearbox_options = [
        "Автомат",
        "Ручна / Механіка",
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
        title = title[:year_match.start()].strip()

    parts = title.split()

    if len(parts) == 0:
        return None, None

    if len(parts) == 1:
        return parts[0], None

    make = parts[0]
    model = parts[1]

    return make, model


def get_search_page(page):
    params = {
        "search_type": 1,
        "category": 1,
        "abroad": 0,
        "customs_cleared": 1,
        "page": page,
    }

    response = requests.get(
        SEARCH_URL,
        params=params,
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


def parse_listing_card(card):
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
    gearbox_text = next((value for value in feature_values if extract_gearbox(value)), full_text)

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
        "url": url,
        "raw_text": full_text,
    }


def parse_search_page(html):
    soup = BeautifulSoup(html, "html.parser")
    cards = find_listing_cards(soup)

    listings = []

    for card in cards:
        item = parse_listing_card(card)

        if item is None:
            continue

        if item["price_usd"] is None:
            continue

        if item["year"] is None:
            continue

        if item["make"] is None:
            continue

        listings.append(item)

    return listings


def load_existing_data(output_path):
    if not os.path.exists(output_path):
        return [], set()

    try:
        df = pd.read_csv(output_path)

        records = df.to_dict("records")
        seen_urls = set(df["url"].dropna().tolist()) if "url" in df.columns else set()

        print(f"Loaded existing file with {len(records)} records.")
        return records, seen_urls

    except Exception:
        return [], set()


def save_data(records, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["url"])

    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def scrape_autoria(
    max_listings=MAX_LISTINGS,
    start_page=0,
    output_path=OUTPUT_PATH,
    min_delay=1.5,
    max_delay=3.5,
):
    all_listings, seen_urls = load_existing_data(output_path)

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
                print("No listings found. The page structure may have changed or scraping was blocked.")
                break

            new_items = 0

            for item in listings:
                if item["url"] in seen_urls:
                    continue

                seen_urls.add(item["url"])
                all_listings.append(item)
                new_items += 1
                progress.update(1)

                if len(all_listings) >= max_listings:
                    break

            print(f"Added {new_items} new listings. Total saved: {len(all_listings)}")

            save_data(all_listings, output_path)

            page += 1

            delay = random.uniform(min_delay, max_delay)
            time.sleep(delay)

        except requests.exceptions.HTTPError as error:
            print(f"HTTP error on page {page}: {error}")
            save_data(all_listings, output_path)
            time.sleep(10)
            page += 1

        except requests.exceptions.RequestException as error:
            print(f"Request error on page {page}: {error}")
            save_data(all_listings, output_path)
            time.sleep(10)
            page += 1

        except KeyboardInterrupt:
            print("\nScraping stopped by user.")
            save_data(all_listings, output_path)
            break

        except Exception as error:
            print(f"Unexpected error on page {page}: {error}")
            save_data(all_listings, output_path)
            time.sleep(10)
            page += 1

    progress.close()

    save_data(all_listings, output_path)

    df = pd.DataFrame(all_listings)
    df = df.drop_duplicates(subset=["url"])

    print("\nDone.")
    print(f"Saved {len(df)} listings to {output_path}")

    return df


if __name__ == "__main__":
    scrape_autoria(
        max_listings=10_000,
        start_page=0,
        output_path="data/autoria_listings.csv",
        min_delay=1.5,
        max_delay=3.5,
    )
