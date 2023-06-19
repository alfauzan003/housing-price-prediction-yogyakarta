import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib

# Function to check if a file exists
def file_exists(filename):
    try:
        with open(filename):
            return True
    except FileNotFoundError:
        return False

# Check if the preprocessed data is already available
data_cache_file = "cache/data_cache.joblib"
if file_exists(data_cache_file):
    data, X_encoded, y = joblib.load(data_cache_file)
else:
    # Read the CSV file if data is not cached
    data = pd.read_csv("data/rumahcom_clean.csv")

    # Prepare data
    X = data[["lokasi", "luas_bangunan", "luas_tanah", "kamar", "kamar_mandi", "listrik", "interior", "sertifikat", "parkir"]]
    y = data["harga"]

    # Encode categorical data
    X_encoded = pd.get_dummies(X, columns=["lokasi", "interior", "sertifikat"])

    # Cache the preprocessed data
    joblib.dump((data, X_encoded, y), data_cache_file)

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Check if the trained model is already available
rf_model_cache_file = "cache/rf_model_cache.joblib"
xgb_model_cache_file = "cache/xgb_model_cache.joblib"
if file_exists(rf_model_cache_file):
    best_rf_model = joblib.load(rf_model_cache_file)
else:
    # Perform hyperparameter tuning if the model is not cached
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=10)
    grid_search.fit(X_train, y_train)

    best_rf_model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)

    # Cache the trained model
    joblib.dump(best_rf_model, rf_model_cache_file)

if file_exists(xgb_model_cache_file):
    best_xgb_model = joblib.load(xgb_model_cache_file)
else:
    # Perform hyperparameter tuning for XGBRegressor if the model is not cached
    param_grid_xgb = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.1, 0.01],
        "gamma": [0, 0.1],
    }

    grid_search_xgb = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid_xgb, cv=10)
    grid_search_xgb.fit(X_train, y_train)

    best_xgb_model = grid_search_xgb.best_estimator_
    print("Best parameters for XGBRegressor:", grid_search_xgb.best_params_)

    # Cache the trained XGBRegressor model
    joblib.dump(best_xgb_model, xgb_model_cache_file)

# Predict using the combined model
y_train_pred_rf = best_rf_model.predict(X_train)
y_train_pred_xgb = best_xgb_model.predict(X_train)

X_train_combined = pd.DataFrame({
    "RF_Prediction": y_train_pred_rf,
    "XGB_Prediction": y_train_pred_xgb
})

y_test_pred_rf = best_rf_model.predict(X_test)
y_test_pred_xgb = best_xgb_model.predict(X_test)

X_test_combined = pd.DataFrame({
    "RF_Prediction": y_test_pred_rf,
    "XGB_Prediction": y_test_pred_xgb
})

# Train a meta-model (RandomForestRegressor) on the combined predictions
meta_model = RandomForestRegressor(random_state=42)
meta_model.fit(X_train_combined, y_train)

# Evaluate the meta-model
train_rmse = mean_squared_error(y_train, meta_model.predict(X_train_combined), squared=False)
train_r2 = r2_score(y_train, meta_model.predict(X_train_combined))
train_mae = mean_absolute_error(y_train, meta_model.predict(X_train_combined))
test_rmse = mean_squared_error(y_test, meta_model.predict(X_test_combined), squared=False)
test_r2 = r2_score(y_test, meta_model.predict(X_test_combined))
test_mae = mean_absolute_error(y_test, meta_model.predict(X_test_combined))

print("Combined Model:")
print("Training RMSE:", train_rmse)
print("Training R2 Score:", train_r2)
print("Training MAE:", train_mae)
print("Testing RMSE:", test_rmse)
print("Testing R2 Score:", test_r2)
print("Testing MAE:", test_mae)


# Predict house price
new_data = pd.DataFrame({
    "lokasi_Sleman": [1],
    "lokasi_Yogyakarta": [0],
    "lokasi_Kulon Progo": [0],
    "lokasi_Bantul": [0],
    "lokasi_Wates": [0],
    "luas_bangunan": [90],
    "luas_tanah": [100],
    "kamar": [3],
    "kamar_mandi": [2],
    "listrik": [1300],
    "interior_Tak Berperabot": [1],
    "interior_Lengkap": [0],
    "interior_Sebagian": [0],
    "parkir": [2],
    "sertifikat_SHM - Sertifikat Hak Milik": [1],
    "sertifikat_SHGB - Hak Guna Bangunan": [0],
    "sertifikat_Sertifikat Belum Pecah": [0],
})

# Reorder the columns to match the order used during training
new_data = new_data[X_encoded.columns]

new_data_pred_rf = best_rf_model.predict(new_data)
new_data_pred_xgb = best_xgb_model.predict(new_data)

new_data_combined = pd.DataFrame({
    "RF_Prediction": new_data_pred_rf,
    "XGB_Prediction": new_data_pred_xgb
})

new_data_pred = meta_model.predict(new_data_combined)
print("Predicted house price:", new_data_pred[0])