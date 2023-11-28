from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import os
import joblib
from flask_wtf import FlaskForm
from wtforms import SelectField
import json


app = Flask(__name__)
app.secret_key = "your_secret_key"  # Change this to a secret key


def find_best_models(mae_dict):
    best_models = {}
    for key, value in mae_dict.items():
        subject, model = key.rsplit("_", 1)
        if subject not in best_models or value < best_models[subject]["mae"]:
            best_models[subject] = {"model": model, "mae": value}

    return best_models


# Åbn JSON-filen
with open("../mae_values.json", "r") as f:
    mae_values = json.load(f)

# Find den bedste model for hvert fag
best_models = find_best_models(mae_values)

# Udskriv resultaterne
for subject, info in best_models.items():
    print(subject, info)


rf_math = joblib.load("../rf_math_model.pkl")
rf_reading = joblib.load("../rf_reading_model.pkl")
rf_writing = joblib.load("../rf_writing_model.pkl")
svm_math = joblib.load("../svm_math_model.pkl")
svm_reading = joblib.load("../svm_reading_model.pkl")
svm_writing = joblib.load("../svm_writing_model.pkl")
mlp_math = joblib.load("../mlp_math_model.pkl")
mlp_reading = joblib.load("../mlp_reading_model.pkl")
mlp_writing = joblib.load("../mlp_writing_model.pkl")

datapath = os.path.join("../../Journal-04", "", "")
dataset = pd.read_csv(datapath + "StudentsPerformance.csv")

# Preprocess the dataset
X = dataset[["gender", "parental level of education", "test preparation course"]]

X = pd.get_dummies(
    X, columns=["gender", "parental level of education", "test preparation course"]
)


@app.route("/", methods=["GET", "POST"])
def home():
    math_prediction = None
    reading_prediction = None
    writing_prediction = None
    svm_math_prediction = None
    svm_reading_prediction = None
    svm_writing_prediction = None
    mlp_math_prediction = None
    mlp_reading_prediction = None
    mlp_writing_prediction = None
    rf_math_prediction = None
    rf_reading_prediction = None
    rf_writing_prediction = None

    if request.method == "POST":
        # Get user input data from the form
        gender = request.form.get("gender")
        education = request.form.get("education")
        prep_course = request.form.get("prep_course")
        model = request.form.get("model")

        # Convert user input into a DataFrame
        user_data = {
            "gender_female": [1 if gender == "female" else 0],
            "gender_male": [1 if gender == "male" else 0],
            f"parental level of education_{education}": [1],
            "test preparation course_completed": [
                1 if prep_course == "completed" else 0
            ],
        }
        user_df = pd.DataFrame(user_data)
        user_df = user_df.reindex(columns=X.columns, fill_value=0)

        # Perform predictions based on the selected model
        if model == "SVM" or model == "All":
            svm_math_prediction = svm_math.predict(user_df)
            svm_reading_prediction = svm_reading.predict(user_df)
            svm_writing_prediction = svm_writing.predict(user_df)
        else:
            svm_math_prediction = svm_reading_prediction = svm_writing_prediction = None

        if model == "MLP" or model == "All":
            mlp_math_prediction = mlp_math.predict(user_df)
            mlp_reading_prediction = mlp_reading.predict(user_df)
            mlp_writing_prediction = mlp_writing.predict(user_df)
        else:
            mlp_math_prediction = mlp_reading_prediction = mlp_writing_prediction = None

        if model == "RF" or model == "All":
            rf_math_prediction = rf_math.predict(user_df)
            rf_reading_prediction = rf_reading.predict(user_df)
            rf_writing_prediction = rf_writing.predict(user_df)
        else:
            rf_math_prediction = rf_reading_prediction = rf_writing_prediction = None

    return render_template(
        "index.html",
        svm_math_prediction=svm_math_prediction,
        svm_reading_prediction=svm_reading_prediction,
        svm_writing_prediction=svm_writing_prediction,
        mlp_math_prediction=mlp_math_prediction,
        mlp_reading_prediction=mlp_reading_prediction,
        mlp_writing_prediction=mlp_writing_prediction,
        rf_math_prediction=rf_math_prediction,
        rf_reading_prediction=rf_reading_prediction,
        rf_writing_prediction=rf_writing_prediction,
        # Overfør MAE-værdier til skabelonen
        svm_math_mae=mae_values.get("mae_math_SVM"),
        svm_reading_mae=mae_values.get("mae_reading_SVM"),
        svm_writing_mae=mae_values.get("mae_writing_SVM"),
        mlp_math_mae=mae_values.get("mae_math_mlp"),
        mlp_reading_mae=mae_values.get("mae_reading_mlp"),
        mlp_writing_mae=mae_values.get("mae_writing_mlp"),
        rf_math_mae=mae_values.get("mae_math_rf"),
        rf_reading_mae=mae_values.get("mae_reading_rf"),
        rf_writing_mae=mae_values.get("mae_writing_rf"),
        best_models=best_models,
    )


if __name__ == "__main__":
    app.run(debug=True)
