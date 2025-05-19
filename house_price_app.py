from flask import Flask, render_template_string, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

app = Flask(__name__)

# Model and scaler filenames
MODEL_FILE = 'house_price_model.pkl'
SCALER_FILE = 'scaler.pkl'

# Location adjustment factors
location_factors = {
    "metro_city": 2.0,
    "inner_city": 1.5,
    "outer_city": 1,
    "tier_2_city": 0.8,
    "small_town": 0.7,
    "village": 0.6,
    "remote_village": 0.5
}

# Features list
features = ['BedroomAbvGr', 'HouseAge', 'HasFurniture', 'HasGarage', 'HasBasement', 'HasHomeGarden', 'HasQualityKitchen']

def train_and_save_model():
    # Load Ames Housing dataset
    data_url = "https://raw.githubusercontent.com/SrikanthVelpuri/House-Prices-Advanced-Regression-Techniques/master/train.csv"
    df = pd.read_csv(data_url)

    # Feature engineering (match amenities)
    amenities = {
        "HasFurniture": "Fireplaces",
        "HasGarage": "GarageCars",
        "HasBasement": "TotalBsmtSF",
        "HasHomeGarden": "PoolQC",
        "HasQualityKitchen": "KitchenQual"
    }
    for new_col, col in amenities.items():
        if col not in df.columns:
            continue
        if col == "KitchenQual":
            df[new_col] = df[col].apply(lambda x: 1 if x in ['Gd', 'Ex'] else 0)
        elif col == "PoolQC":
            df[new_col] = df[col].notna().astype(int)
        elif df[col].dtype == 'object':
            df[new_col] = df[col].notna().astype(int)
        else:
            df[new_col] = df[col].apply(lambda x: 1 if x > 0 else 0)

    # Compute house age
    df['HouseAge'] = 2025 - df['YearBuilt']

    # Define features and target
    df = df.dropna(subset=features + ['SalePrice'])  # Drop rows with missing values
    X = df[features]
    y = df['SalePrice']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model with cross-validation and hyperparameter tuning
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    # Evaluate model metrics
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Save model and scaler
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    return rmse, r2, mae

# Load model and scaler (train if not available)
if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE)):
    rmse, r2, mae = train_and_save_model()
else:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    # Loading dataset again only to calculate evaluation metrics (to display on evaluate page)
    data_url = "https://raw.githubusercontent.com/SrikanthVelpuri/House-Prices-Advanced-Regression-Techniques/master/train.csv"
    df = pd.read_csv(data_url)
    amenities = {
        "HasFurniture": "Fireplaces",
        "HasGarage": "GarageCars",
        "HasBasement": "TotalBsmtSF",
        "HasHomeGarden": "PoolQC",
        "HasQualityKitchen": "KitchenQual"
    }
    for new_col, col in amenities.items():
        if col not in df.columns:
            continue
        if col == "KitchenQual":
            df[new_col] = df[col].apply(lambda x: 1 if x in ['Gd', 'Ex'] else 0)
        elif col == "PoolQC":
            df[new_col] = df[col].notna().astype(int)
        elif df[col].dtype == 'object':
            df[new_col] = df[col].notna().astype(int)
        else:
            df[new_col] = df[col].apply(lambda x: 1 if x > 0 else 0)
    df['HouseAge'] = 2025 - df['YearBuilt']
    df = df.dropna(subset=features + ['SalePrice'])
    X = df[features]
    y = df['SalePrice']
    scaler = joblib.load(SCALER_FILE)
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

# Templates for the pages
menu_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Menu</title>
  <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f4f8; color: #333; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; margin:0; }
    h1 { color: #0056b3; }
    .btn-container { display: flex; gap: 20px; margin-top: 30px; }
    button { padding: 15px 30px; font-size: 1.2rem; border: none; border-radius: 8px; cursor: pointer; background: linear-gradient(45deg, #1e90ff, #0056b3); color: white; transition: background 0.3s ease; }
    button:hover { background: linear-gradient(45deg, #0056b3, #003a75); }
  </style>
</head>
<body>
  <h1>Menu</h1>
  <div class="btn-container">
    <button onclick="location.href='{{ url_for('predict') }}'">Predict</button>
    <button onclick="location.href='{{ url_for('evaluate') }}'">Evaluate</button>
    <button onclick="location.href='{{ url_for('graph') }}'">Graph</button>
  </div>
</body>
</html>
"""

evaluate_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Model Evaluation</title>
  <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #fff; color: #333; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; margin:0; }
    h1 { color: #d9534f; }
    p { font-size: 1.3rem; margin: 10px 0; }
    button { margin-top: 30px; padding: 12px 28px; font-size: 1.1rem; border-radius: 8px; border: none; cursor: pointer; background-color: #0275d8; color: white; }
    button:hover { background-color: #025aa5; }
  </style>
</head>
<body>
  <h1>--- Model Evaluation ---</h1>
  <p>📉 RMSE: {{ rmse | round(0) | intcomma }}</p>
  <p>📈 R² Score: {{ r2 | round(2) }}</p>
  <p>📊 MAE: {{ mae | round(0) | intcomma }}</p>
  <button onclick="location.href='{{ url_for('menu') }}'">Go Back</button>
</body>
</html>
"""

graph_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Graph</title>
  <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #fff; color: #333; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; min-height: 100vh; margin:0; padding-top: 20px;}
    h1 { color: #27ae60; margin-bottom: 20px; }
    .imgs { display: flex; gap: 20px; }
    img { width: 300px; height: auto; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
    button { margin-top: 40px; padding: 12px 28px; font-size: 1.1rem; border-radius: 8px; border: none; cursor: pointer; background-color: #5bc0de; color: white; }
    button:hover { background-color: #31b0d5; }
  </style>
</head>
<body>
  <h1>Graphs</h1>
  <div class="imgs">
    <img src="{{ url_for('static', filename='pic1.jpg') }}" alt="Picture 1" />
    <img src="{{ url_for('static', filename='pic2.jpg') }}" alt="Picture 2" />
  </div>
  <button onclick="location.href='{{ url_for('menu') }}'">Go Back</button>
</body>
</html>
"""

predict_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Predict House Price</title>
  <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f7f9fc; color: #222; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding-top: 40px; margin: 0;}
    h1 { color: #004085; margin-bottom: 25px; }
    form { background: white; padding: 25px 30px; border-radius: 12px; box-shadow: 0 6px 15px rgba(0,0,0,0.1); width: 320px; }
    label { display: block; margin-bottom: 6px; font-weight: 600; }
    input[type=number], select { width: 100%; padding: 8px 10px; margin-bottom: 18px; border-radius: 6px; border: 1px solid #ccc; font-size: 1rem; }
    button { width: 100%; padding: 12px 0; background-color: #28a745; color: white; border: none; font-size: 1.1rem; border-radius: 8px; cursor: pointer; }
    button:hover { background-color: #218838; }
    .result { background: #e9f7ef; border: 1px solid #c3e6cb; color: #155724; margin-top: 20px; padding: 15px; border-radius: 8px; font-weight: 700; font-size: 1.2rem; }
    .back-button { margin-top: 30px; padding: 10px 24px; background: #007bff; color: white; border: none; border-radius: 8px; cursor: pointer; text-align: center; font-size: 1rem; }
    .back-button:hover { background: #0056b3; }
  </style>
</head>
<body>
  <h1>Predict House Price</h1>
  <form method="post" action="{{ url_for('predict') }}">
    <label for="bedrooms">Number of Bedrooms:</label>
    <input type="number" id="bedrooms" name="bedrooms" min="0" required value="{{ request.form.bedrooms if request.method=='POST' else '' }}">

    <label for="house_age">House Age (years):</label>
    <input type="number" id="house_age" name="house_age" min="0" required value="{{ request.form.house_age if request.method=='POST' else '' }}">

    <label for="has_furniture">Include Furniture:</label>
    <select id="has_furniture" name="has_furniture" required>
      <option value="" disabled selected>Select</option>
      <option value="1" {% if request.form.get('has_furniture') == '1' %}selected{% endif %}>Yes</option>
      <option value="0" {% if request.form.get('has_furniture') == '0' %}selected{% endif %}>No</option>
    </select>

    <label for="has_garage">Include Garage:</label>
    <select id="has_garage" name="has_garage" required>
      <option value="" disabled selected>Select</option>
      <option value="1" {% if request.form.get('has_garage') == '1' %}selected{% endif %}>Yes</option>
      <option value="0" {% if request.form.get('has_garage') == '0' %}selected{% endif %}>No</option>
    </select>

    <label for="has_basement">Include Basement:</label>
    <select id="has_basement" name="has_basement" required>
      <option value="" disabled selected>Select</option>
      <option value="1" {% if request.form.get('has_basement') == '1' %}selected{% endif %}>Yes</option>
      <option value="0" {% if request.form.get('has_basement') == '0' %}selected{% endif %}>No</option>
    </select>

    <label for="has_home_garden">Include Home Garden:</label>
    <select id="has_home_garden" name="has_home_garden" required>
      <option value="" disabled selected>Select</option>
      <option value="1" {% if request.form.get('has_home_garden') == '1' %}selected{% endif %}>Yes</option>
      <option value="0" {% if request.form.get('has_home_garden') == '0' %}selected{% endif %}>No</option>
    </select>

    <label for="has_quality_kitchen">Include Quality Kitchen:</label>
    <select id="has_quality_kitchen" name="has_quality_kitchen" required>
      <option value="" disabled selected>Select</option>
      <option value="1" {% if request.form.get('has_quality_kitchen') == '1' %}selected{% endif %}>Yes</option>
      <option value="0" {% if request.form.get('has_quality_kitchen') == '0' %}selected{% endif %}>No</option>
    </select>

    <label for="location_type">Indian Location Type:</label>
    <select id="location_type" name="location_type" required>
      <option value="" disabled selected>Select</option>
      {% for key in location_factors.keys() %}
      <option value="{{ key }}" {% if request.form.get('location_type') == key %}selected{% endif %}>{{ key.replace('_', ' ').title() }}</option>
      {% endfor %}
    </select>

    <label for="exchange_rate">USD to INR Exchange Rate:</label>
    <input type="number" id="exchange_rate" name="exchange_rate" min="0" step="0.01" required value="{{ request.form.exchange_rate if request.method=='POST' else '' }}">

    <button type="submit">Predict</button>
  </form>

  {% if prediction %}
  <div class="result">
    ✅ Predicted house price: ₹{{ "{:,.0f}".format(prediction) }}
  </div>
  {% endif %}

  <button class="back-button" onclick="location.href='{{ url_for('menu') }}'">Go Back</button>
</body>
</html>
"""

# Filter for thousands separator formatting
def intcomma_filter(value):
    return f"{value:,}"

app.jinja_env.filters['intcomma'] = intcomma_filter

@app.route('/')
@app.route('/menu')
def menu():
    return render_template_string(menu_template)

@app.route('/evaluate')
def evaluate():
    return render_template_string(evaluate_template, rmse=rmse, r2=r2, mae=mae)

@app.route('/graph')
def graph():
    return render_template_string(graph_template)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            bedrooms = int(request.form['bedrooms'])
            house_age = int(request.form['house_age'])
            has_furniture = int(request.form['has_furniture'])
            has_garage = int(request.form['has_garage'])
            has_basement = int(request.form['has_basement'])
            has_home_garden = int(request.form['has_home_garden'])
            has_quality_kitchen = int(request.form['has_quality_kitchen'])
            location_type = request.form['location_type']
            exchange_rate = float(request.form['exchange_rate'])

            # Validate location type
            if location_type not in location_factors:
                return render_template_string(predict_template, prediction=None, location_factors=location_factors, error="Invalid location type.")

            location_factor = location_factors[location_type]

            # Prepare input data
            input_data = pd.DataFrame([[bedrooms, house_age, has_furniture, has_garage,
                                        has_basement, has_home_garden, has_quality_kitchen]],
                                      columns=features)

            # Scale input data
            input_data_scaled = scaler.transform(input_data)

            # Predict house price in USD
            predicted_price_usd = model.predict(input_data_scaled)[0]

            # Convert and adjust for location
            predicted_price_inr = np.round(predicted_price_usd * exchange_rate * location_factor * 0.1)

            prediction = int(predicted_price_inr)

        except Exception as e:
            # Could log exception here
            prediction = None

    return render_template_string(predict_template, prediction=prediction, location_factors=location_factors)


if __name__ == '__main__':
    app.run(debug=True)

