# Tugas Akhir Data Science
Predict the appropriate price for a house based on its specifications, such as square footage, number of rooms, location, etc. The model used is a combination of Random Forest and XGB Regressor.

## Dataset Preparation

Save the clean dataset as rumahcom_clean.csv in the project directory.

## Model Training and Evaluation

Train the model

`python model.py`

The trained model will be saved as model_cache.joblib.

## Predicting House Prices

To predict house prices for new data:

Edit the new_data dictionary in the model.py file with the desired values.

Run the script again:

`python model.py`

## Accuracy

The model still lack on accuracy, and will be update soon.

## Running on Web

To run this project on website, clone this repository and run:

`cd web`

`python app.py`

or

`--app app.py --debug run`

After server running, open https://localhost:5000 on your browser
