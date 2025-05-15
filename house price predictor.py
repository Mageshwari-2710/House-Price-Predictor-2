import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

# Load Ames Housing dataset
data_url = "https://raw.githubusercontent.com/SrikanthVelpuri/House-Prices-Advanced-Regression-Techniques/master/train.csv"
df = pd.read_csv(data_url)
print("✔ Loaded Ames Housing dataset.")

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
features = ['BedroomAbvGr', 'HouseAge', 'HasFurniture', 'HasGarage', 'HasBasement', 'HasHomeGarden', 'HasQualityKitchen']
df = df.dropna(subset=features + ['SalePrice'])  # Drop rows with missing values
X = df[features]
y = df['SalePrice']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model with cross-validation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
print("✔ Trained model with updated features.")

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

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

# Save the model
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Helper for validated integer input
def get_int_input(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")

# Menu system
def main_menu():
    while True:
        print("\n=======================")
        print("🏠 House Price Predictor")
        print("=======================")
        print("1. Predict House Price")
        print("2. Evaluate Model")
        print("3. Show Feature Importance")
        print("4. Plot Actual vs Predicted")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            predict_house_price()
        elif choice == "2":
            evaluate_model()
        elif choice == "3":
            show_feature_importance()
        elif choice == "4":
            plot_actual_vs_predicted()
        elif choice == "5":
            print("👋 Exiting program.")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def predict_house_price():
    print("\n--- Predicting House Price ---")
    base_bedrooms = get_int_input("Enter number of bedrooms: ")
    house_age = get_int_input("Enter house age in years: ")
    has_furniture = get_int_input("Include Furniture? (1 for yes, 0 for no): ")
    has_garage = get_int_input("Include Garage? (1 for yes, 0 for no): ")
    has_basement = get_int_input("Include Basement? (1 for yes, 0 for no): ")
    has_home_garden = get_int_input("Include Home Garden? (1 for yes, 0 for no): ")
    has_quality_kitchen = get_int_input("Include Quality Kitchen? (1 for yes, 0 for no): ")

    print("\nSelect Indian Location Type:")
    for loc in location_factors:
        print(f"  - {loc.replace('_', ' ').title()}")

    while True:
        location_input = input("Enter location type: ").strip().lower().replace(" ", "_")
        if location_input in location_factors:
            location_factor = location_factors[location_input]
            break
        else:
            print("❌ Invalid location. Try again.")

    while True:
        try:
            exchange_rate = float(input("Enter USD to INR exchange rate (e.g., 85): "))
            break
        except ValueError:
            print("❌ Invalid rate. Please enter a number.")

    input_data = pd.DataFrame([[base_bedrooms, house_age, has_furniture, has_garage,
                                has_basement, has_home_garden, has_quality_kitchen]],
                              columns=features)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    predicted_price = model.predict(input_data_scaled)
    predicted_price_inr = np.round(predicted_price[0] * exchange_rate * location_factor * 0.1)
    print(f"\n✅ Predicted house price: ₹{predicted_price_inr:,.0f}")

def evaluate_model():
    print("\n--- Model Evaluation ---")
    print(f"📉 RMSE: {rmse:,.0f}")
    print(f"📈 R² Score: {r2:.2f}")
    print(f"📊 MAE: {mae:,.0f}")

def show_feature_importance():
    print("\n--- Feature Importance ---")
    importances = model.feature_importances_
    feature_names = features
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances, color='teal')
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted():
    print("\n--- Actual vs Predicted Plot ---")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title("Actual vs Predicted")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Start
main_menu()
