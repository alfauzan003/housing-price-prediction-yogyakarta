import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Function to check if a file exists


def file_exists(filename):
   try:
      with open(filename):
         return True
   except FileNotFoundError:
      return False


# Check if the preprocessed data is already available
data_cache_file = "data_cache.joblib"
if file_exists(data_cache_file):
   data, X_encoded, y = joblib.load(data_cache_file)
else:
   # Read the CSV file if data is not cached
   data = pd.read_csv("rumahcom_clean.csv")

   # Prepare data
   X = data[["lokasi", "luas_bangunan", "luas_tanah", "kamar",
             "kamar_mandi", "listrik", "interior", "sertifikat", "parkir"]]
   y = data["harga"]

   # Encode categorical data
   X_encoded = pd.get_dummies(X, columns=["lokasi", "interior", "sertifikat"])

   # Cache the preprocessed data
   joblib.dump((data, X_encoded, y), data_cache_file)

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42)

# Check if the trained model is already available
model_cache_file = "model_cache.joblib"
if file_exists(model_cache_file):
   best_rf_model = joblib.load(model_cache_file)
else:
   # Perform hyperparameter tuning if the model is not cached
   param_grid = {
       "n_estimators": [100, 200],
       "max_depth": [None, 10],
       "min_samples_split": [2, 5],
       "min_samples_leaf": [1, 2],
   }

   grid_search = GridSearchCV(RandomForestRegressor(
       random_state=42), param_grid, cv=5)
   grid_search.fit(X_train, y_train)

   best_rf_model = grid_search.best_estimator_
   print("Best parameters:", grid_search.best_params_)

   # Cache the trained model
   joblib.dump(best_rf_model, model_cache_file)

# Evaluate Random Forest model
y_train_pred = best_rf_model.predict(X_train)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
print("Training RMSE:", train_rmse)
print("Training R2 Score:", train_r2)

y_test_pred = best_rf_model.predict(X_test)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)
print("Testing RMSE:", test_rmse)
print("Testing R2 Score:", test_r2)

# Predict house price
new_data = pd.DataFrame({
    "lokasi_Sleman": [1],
    "lokasi_Yogyakarta": [0],
    "lokasi_Kulon Progo": [0],
    "lokasi_Bantul": [0],
    "lokasi_Wates": [0],
    "luas_bangunan": [179],
    "luas_tanah": [212],
    "kamar": [10],
    "kamar_mandi": [10],
    "listrik": [3500],
    "interior_Tak Berperabot": [1],
    "interior_Lengkap": [0],
    "interior_Sebagian": [0],
    "parkir": [6],
    "sertifikat_SHM - Sertifikat Hak Milik": [1],
    "sertifikat_SHGB - Hak Guna Bangunan": [0],
    "sertifikat_Sertifikat Belum Pecah": [0],
})

new_data = new_data[X_encoded.columns.tolist()]

new_data_pred = best_rf_model.predict(new_data)
print("Predicted house price:", new_data_pred)
