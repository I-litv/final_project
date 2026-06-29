import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = PROJECT_DIR / "used_cars.csv"
SCRAPER_PATH = PROJECT_DIR / "scraper" / "scrape_autoria.py"
DEFAULT_MAX_LISTINGS = 20_000


# ============================================================
# CHANGE COLUMN NAMES HERE IF YOUR CSV USES DIFFERENT NAMES
# Left side: names used inside this project.
# Right side: names from your CSV file.
# ============================================================
COLUMN_NAMES = {
    "make": "make",
    "model": "model",
    "year": "year",
    "mileage_km": "mileage_km",
    "price_usd": "price_usd",
}


FEATURE_COLUMNS = ["make", "model", "year", "mileage_km"]
TARGET_COLUMN = "price_usd"
CATEGORICAL_COLUMNS = ["make", "model"]
NUMERIC_COLUMNS = ["year", "mileage_km"]


def run_scraper_before_training(
    csv_path,
    max_listings,
    start_page,
    min_delay,
    max_delay,
    allow_partial_scrape,
):
    """Run the scraper and wait for it to finish before training starts."""
    if not SCRAPER_PATH.exists():
        raise FileNotFoundError(
            "Scraper file was not found at "
            f"{SCRAPER_PATH}. Make sure scraper/scrape_autoria.py exists."
        )

    spec = importlib.util.spec_from_file_location("scrape_autoria", SCRAPER_PATH)
    scraper_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scraper_module)

    print("Starting scraper before model training...")
    scraped_data = scraper_module.scrape_autoria(
        max_listings=max_listings,
        start_page=start_page,
        output_path=csv_path,
        min_delay=min_delay,
        max_delay=max_delay,
    )

    scraped_count = len(scraped_data)
    if scraped_count < max_listings and not allow_partial_scrape:
        raise RuntimeError(
            f"Scraper collected {scraped_count} listings, but {max_listings} "
            "were requested. Training stopped so the model does not use a "
            "partially refreshed dataset. Run again later, lower "
            "--max-listings, or pass --allow-partial-scrape."
        )

    print(f"Scraper finished with {scraped_count} listings.")


def normalize_text(value):
    """Convert text to lowercase and remove extra spaces."""
    return str(value).lower().strip()


def load_and_clean_data(csv_path):
    """Load the CSV file and keep only valid rows for model training."""
    raw_data = pd.read_csv(csv_path)
    cars_on_sale_count = len(raw_data)

    missing_columns = [
        csv_column
        for csv_column in COLUMN_NAMES.values()
        if csv_column not in raw_data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing columns in CSV: "
            + ", ".join(missing_columns)
            + ". Update COLUMN_NAMES in train_model.py if needed."
        )

    data = raw_data[
        [
            COLUMN_NAMES["make"],
            COLUMN_NAMES["model"],
            COLUMN_NAMES["year"],
            COLUMN_NAMES["mileage_km"],
            COLUMN_NAMES["price_usd"],
        ]
    ].copy()
    data.columns = ["make", "model", "year", "mileage_km", "price_usd"]

    important_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    data = data.dropna(subset=important_columns)

    data["make"] = data["make"].apply(normalize_text)
    data["model"] = data["model"].apply(normalize_text)

    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["mileage_km"] = pd.to_numeric(data["mileage_km"], errors="coerce")
    data["price_usd"] = pd.to_numeric(data["price_usd"], errors="coerce")
    data = data.dropna(subset=important_columns)

    current_year = datetime.now().year
    data = data[
        (data["make"] != "")
        & (data["model"] != "")
        & (data["year"] >= 1990)
        & (data["year"] <= current_year)
        & (data["mileage_km"] >= 0)
        & (data["price_usd"] > 0)
    ].copy()

    data["year"] = data["year"].astype(int)
    data["mileage_km"] = data["mileage_km"].astype(float)
    data["price_usd"] = data["price_usd"].astype(float)

    return data, cars_on_sale_count


def create_pipeline(regression_model):
    """Create one complete model pipeline with preprocessing and regression."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regression_model),
        ]
    )

    return pipeline


def evaluate_model(model, x_train, x_test, y_train, y_test):
    """Train one model and return common regression metrics."""
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2_score": round(float(r2), 4),
    }


def train_and_compare_models(data):
    """Train several beginner-friendly models and choose the best one by MAE."""
    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    print("Splitting data into training and test sets...", flush=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    knn_neighbors = min(5, len(x_train))
    model_options = {
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=knn_neighbors),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
    }

    all_metrics = {}
    best_model_name = None
    best_mae = float("inf")

    for model_name, regression_model in model_options.items():
        print(f"Training {model_name}...", flush=True)
        pipeline = create_pipeline(regression_model)
        metrics = evaluate_model(pipeline, x_train, x_test, y_train, y_test)
        all_metrics[model_name] = metrics
        print(
            f"{model_name} complete: "
            f"MAE={metrics['mae']}, "
            f"RMSE={metrics['rmse']}, "
            f"R2={metrics['r2_score']}",
            flush=True,
        )

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_model_name = model_name

    print(f"Training final {best_model_name} on all cleaned rows...", flush=True)
    best_pipeline = create_pipeline(clone(model_options[best_model_name]))
    best_pipeline.fit(x, y)

    return best_pipeline, best_model_name, all_metrics


def save_metrics(
    metrics_path,
    csv_path,
    data,
    cars_on_sale_count,
    best_model_name,
    all_metrics,
):
    """Save evaluation results so the Streamlit app can display them."""
    metrics = {
        "project": "Used Car Price Estimator for Ukraine",
        "dataset_path": str(csv_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cars_on_sale_count": int(cars_on_sale_count),
        "rows_after_cleaning": int(len(data)),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "best_model": best_model_name,
        "best_model_metrics": all_metrics[best_model_name],
        "all_model_metrics": all_metrics,
        "selection_note": "Best model selected mainly by lowest MAE.",
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Train a used car price estimator from a CSV dataset."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_CSV_PATH),
        help=(
            "Path to the CSV dataset. By default, the scraper refreshes this "
            "file before training."
        ),
    )
    parser.add_argument(
        "--model-output",
        default="car_price_model.pkl",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--metrics-output",
        default="model_metrics.json",
        help="Where to save model evaluation metrics.",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Train from the existing CSV file without running the scraper first.",
    )
    parser.add_argument(
        "--max-listings",
        type=int,
        default=DEFAULT_MAX_LISTINGS,
        help="How many listings the scraper should collect before training.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=0,
        help="AUTO.RIA search page where scraping should start.",
    )
    parser.add_argument(
        "--scrape-min-delay",
        type=float,
        default=1.5,
        help="Minimum delay between scraper page requests.",
    )
    parser.add_argument(
        "--scrape-max-delay",
        type=float,
        default=3.5,
        help="Maximum delay between scraper page requests.",
    )
    parser.add_argument(
        "--allow-partial-scrape",
        action="store_true",
        help="Allow training even if the scraper collected fewer listings than requested.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    model_output_path = Path(args.model_output)
    metrics_output_path = Path(args.metrics_output)

    if not args.skip_scrape:
        run_scraper_before_training(
            csv_path=csv_path,
            max_listings=args.max_listings,
            start_page=args.start_page,
            min_delay=args.scrape_min_delay,
            max_delay=args.scrape_max_delay,
            allow_partial_scrape=args.allow_partial_scrape,
        )

    print(f"Loading dataset from: {csv_path}", flush=True)
    data, cars_on_sale_count = load_and_clean_data(csv_path)
    print(f"Loaded {cars_on_sale_count} rows from CSV.", flush=True)
    print(f"Rows available after cleaning: {len(data)}", flush=True)
    if len(data) < 10:
        raise ValueError(
            "Not enough valid rows after cleaning. "
            "Please provide at least 10 usable rows."
    )

    best_model, best_model_name, all_metrics = train_and_compare_models(data)

    print(f"Saving model to: {model_output_path}", flush=True)
    joblib.dump(best_model, model_output_path, compress=3)
    print(f"Saving metrics to: {metrics_output_path}", flush=True)
    save_metrics(
        metrics_output_path,
        csv_path,
        data,
        cars_on_sale_count,
        best_model_name,
        all_metrics,
    )

    print("Training complete.")
    print(f"Cars currently on sale in dataset: {cars_on_sale_count}")
    print(f"Rows used after cleaning: {len(data)}")
    print(f"Best model: {best_model_name}")
    print(f"Best MAE: {all_metrics[best_model_name]['mae']}")
    print(f"Best R2 score: {all_metrics[best_model_name]['r2_score']}")
    print(f"Saved model to: {model_output_path}")
    print(f"Saved metrics to: {metrics_output_path}")


if __name__ == "__main__":
    main()
