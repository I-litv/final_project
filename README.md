# Used Car Price Estimator for Ukraine

This is a simple final project for an AI Fundamentals course. It trains a
machine learning model that estimates the average used car price in Ukraine
and shows a simple projected price trend until the end of the current year
based on:

- Car make
- Car model
- Manufacturing year
- Current mileage in kilometers
- Expected monthly mileage in kilometers

The prediction is an educational estimate, not a guaranteed market price.
The future trend is a simple machine-learning projection based mainly on
increasing mileage, not a guaranteed market forecast.

## Project Files

- `train_model.py` - loads the CSV dataset, cleans it, trains several models,
  compares metrics, and saves the best model.
- `app.py` - Streamlit web app for entering car information and getting a
  current price estimate, a projected price table, and a projected price chart.
- `../scraper/scrape_autoria.py` - collects current AUTO.RIA listings and
  saves them as `used_cars.csv` before model training.
- `requirements.txt` - Python packages needed to run the project.
- `model_metrics.json` - generated after training.
- `car_price_model.pkl` - generated after training.

## Expected Dataset Format

The CSV dataset should contain these columns:

```text
make
model
year
mileage_km
price_usd
```

The scraper also saves optional columns such as `listing_date`, `fuel`,
`gearbox`, `city`, `url`, and `raw_text`. The machine learning model only uses
the required columns, but the app uses `listing_date` to estimate how long a
similar active listing has usually been online.

If your dataset uses different column names, open `train_model.py` and edit
this clearly marked section near the top:

```python
COLUMN_NAMES = {
    "make": "make",
    "model": "model",
    "year": "year",
    "mileage_km": "mileage_km",
    "price_usd": "price_usd",
}
```

For example, if your CSV uses `brand` instead of `make`, change it to:

```python
COLUMN_NAMES = {
    "make": "brand",
    "model": "model",
    "year": "year",
    "mileage_km": "mileage_km",
    "price_usd": "price_usd",
}
```

## What The Training Script Does

`train_model.py`:

1. Runs the AUTO.RIA scraper first and waits until it finishes.
2. Saves scraper output as `used_cars.csv`.
3. Loads the CSV file with pandas.
4. Removes rows with missing important values.
5. Converts `year`, `mileage_km`, and `price_usd` to numeric values.
6. Removes impossible values:
   - Year less than 1990
   - Year greater than the current year
   - Mileage below 0
   - Price less than or equal to 0
7. Normalizes `make` and `model` by converting them to lowercase and stripping
   extra spaces.
8. Uses OneHotEncoder for `make` and `model`.
9. Uses `year` and `mileage_km` as numeric features.
10. Trains and compares:
   - Linear Regression
   - KNN Regressor
   - Decision Tree Regressor
   - Random Forest Regressor
11. Evaluates each model with:
   - MAE
   - RMSE
   - R2 score
12. Saves the best model by lowest MAE as `car_price_model.pkl`.
13. Saves metrics as `model_metrics.json`.
14. Saves the number of cars currently on sale in the dataset, based on the
    number of listing rows in the CSV file.

By default, the scraper tries to collect `20,000` listings. Training starts
only after the scraper finishes. If fewer listings are collected, training
stops unless you pass `--allow-partial-scrape`.

## What The Streamlit App Does

`app.py`:

1. Loads the trained model from `car_price_model.pkl`.
2. Lets the user type the car make and model manually.
3. Lets the user enter manufacturing year, current mileage, and expected
   monthly mileage.
4. Predicts the current estimated average price.
5. Creates one future prediction for each month until December of the current
   year.
6. Increases mileage each month by the expected monthly mileage.
7. Shows the projected values in a table.
8. Shows a line chart of predicted price by month.
9. Shows how many car listings are currently in the dataset used for training.
10. After prediction, shows how many cars with the same make, model, and
    manufacturing year are currently present in the dataset.
11. Uses listing publication dates to estimate how long a similar active
    listing has usually been online.

## How To Run

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Train the model. This runs the scraper first, saves fresh data to
`used_cars.csv`, and then trains:

```bash
python3 train_model.py
```

This creates:

```text
car_price_model.pkl
model_metrics.json
```

For a faster local check without scraping, train from the existing CSV:

```bash
python3 train_model.py --skip-scrape used_cars.csv
```

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Example Input In The App

```text
Car make: Toyota
Car model: Corolla
Manufacturing year: 2015
Current mileage in kilometers: 120000
Expected monthly mileage in kilometers: 1000
```

The app will normalize typed text automatically, so `Toyota`, `toyota`, and
` toyota ` are treated the same way.

## Important Note About Typed Make And Model

The app uses typed text inputs instead of dropdowns. The model is configured
with `OneHotEncoder(handle_unknown="ignore")`, so it can still make a
prediction if the user types a make or model that was not present in the
training data. However, that prediction may be less accurate.

## Important Note About Sale Time Estimate

The app estimates sale time from the median age of active listings with a
publication date. This is not confirmed sold-time data, because the dataset
does not include the exact date when a car was sold or removed from the site.
