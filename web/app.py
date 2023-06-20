import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
from flask import Flask, render_template, request

current_dir = os.path.dirname(os.path.abspath(__file__))
template_folder = os.path.join(current_dir)
app = Flask(__name__, template_folder=template_folder)

# Function to check if a file exists
def file_exists(filename):
   try:
      with open(filename):
         return True
   except FileNotFoundError:
      return False


@app.route('/', methods=['GET', 'POST'])
def predict_house_price():
   if request.method == 'POST':
      # Check if the preprocessed data is already available
      data_cache_file = "../cache/data_cache.joblib"
      if file_exists(data_cache_file):
         data, X_encoded, y = joblib.load(data_cache_file)
      else:
         # Read the CSV file if data is not cached
         data = pd.read_csv("../cache/rumahcom_clean.csv")

         # Prepare data
         X = data[["lokasi", "luas_bangunan", "luas_tanah", "kamar", "kamar_mandi", "listrik", "interior", "sertifikat", "parkir"]]
         y = data["harga"]

         # Encode categorical data
         X_encoded = pd.get_dummies(
            X, columns=["lokasi", "interior", "sertifikat"])

         # Cache the preprocessed data
         joblib.dump((data, X_encoded, y), data_cache_file)

      # Split data into training and testing data
      X_train, X_test, y_train, y_test = train_test_split(
         X_encoded, y, test_size=0.2, random_state=42)

      # Check if the trained model is already available
      rf_model_cache_file = "../cache/rf_model_cache.joblib"
      xgb_model_cache_file = "../cache/xgb_model_cache.joblib"
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

         grid_search = GridSearchCV(RandomForestRegressor(
            random_state=42), param_grid, cv=10)
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

         grid_search_xgb = GridSearchCV(xgb.XGBRegressor(
            random_state=42), param_grid_xgb, cv=10)
         grid_search_xgb.fit(X_train, y_train)

         best_xgb_model = grid_search_xgb.best_estimator_
         print("Best parameters for XGBRegressor:",
               grid_search_xgb.best_params_)

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

      # Predict house price
      lokasi = request.form['lokasi']
      luas_bangunan = float(request.form['luas_bangunan'])
      luas_tanah = float(request.form['luas_tanah'])
      kamar = int(request.form['kamar'])
      kamar_mandi = int(request.form['kamar_mandi'])
      listrik = int(request.form['listrik'])
      interior = request.form['interior']
      sertifikat = request.form['sertifikat']
      parkir = int(request.form['parkir'])

      new_data = pd.DataFrame({
         "lokasi_Sleman": [1 if lokasi == 'Sleman' else 0],
         "lokasi_Yogyakarta": [1 if lokasi == 'Yogyakarta' else 0],
         "lokasi_Kulon Progo": [1 if lokasi == 'Kulon Progo' else 0],
         "lokasi_Bantul": [1 if lokasi == 'Bantul' else 0],
         "lokasi_Wates": [1 if lokasi == 'Wates' else 0],
         "luas_bangunan": [luas_bangunan],
         "luas_tanah": [luas_tanah],
         "kamar": [kamar],
         "kamar_mandi": [kamar_mandi],
         "listrik": [listrik],
         "interior_Tak Berperabot": [1 if interior == 'Tak Berperabot' else 0],
         "interior_Lengkap": [1 if interior == 'Lengkap' else 0],
         "interior_Sebagian": [1 if interior == 'Sebagian' else 0],
         "parkir": [parkir],
         "sertifikat_SHM - Sertifikat Hak Milik": [1 if sertifikat == 'SHM - Sertifikat Hak Milik' else 0],
         "sertifikat_SHGB - Hak Guna Bangunan": [1 if sertifikat == 'SHGB - Hak Guna Bangunan' else 0],
         "sertifikat_Sertifikat Belum Pecah": [1 if sertifikat == 'Sertifikat Belum Pecah' else 0],
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
      predicted_price = new_data_pred[0]

      formatted_prediction = "{:,.0f}".format(predicted_price)
      formatted_prediction = formatted_prediction.replace(",", ".")

      return render_template('index.html', predicted_price=formatted_prediction)
   else:
      return render_template('index.html')


if __name__ == '__main__':
   app.run()
