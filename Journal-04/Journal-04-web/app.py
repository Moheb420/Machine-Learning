from flask import Flask, render_template, request
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import os

app = Flask(__name__)

# Load dataset
datapath = os.path.join("../../Journal-04", "", "")
dataset = pd.read_csv(datapath + "StudentsPerformance.csv")

# Preprocess the dataset
X = dataset[['gender', 'parental level of education', 'test preparation course']]
y_math = dataset['math score']
y_reading = dataset['reading score']
y_writing = dataset['writing score']

X = pd.get_dummies(X, columns=['gender', 'parental level of education', 'test preparation course'])

# Train the SVR models
X_train_math, X_test_math, y_math_train, y_math_test = train_test_split(X, y_math, test_size=0.2, random_state=42)
svm_math = SVR(kernel='linear')
svm_math.fit(X_train_math, y_math_train)

X_train_reading, X_test_reading, y_reading_train, y_reading_test = train_test_split(X, y_reading, test_size=0.2, random_state=42)
svm_reading = SVR(kernel='linear')
svm_reading.fit(X_train_reading, y_reading_train)

X_train_writing, X_test_writing, y_writing_train, y_writing_test = train_test_split(X, y_writing, test_size=0.2, random_state=42)
svm_writing = SVR(kernel='linear')
svm_writing.fit(X_train_writing, y_writing_train)

@app.route("/", methods=["GET", "POST"])
def home():
    math_prediction = None
    reading_prediction = None
    writing_prediction = None

    if request.method == "POST":
        # Get user input data from the form
        gender = request.form.get("gender")
        education = request.form.get("education")
        prep_course = request.form.get("prep_course")

        # Convert user input into a DataFrame
        user_data = {
            'gender_female': [1 if gender == "female" else 0],
            'gender_male': [1 if gender == "male" else 0],
            f"parental level of education_{education}": [1],
            'test preparation course_completed': [1 if prep_course == "completed" else 0]
        }
        user_df = pd.DataFrame(user_data)
        user_df = user_df.reindex(columns=X.columns, fill_value=0)

        # Perform predictions for math, reading, and writing scores
        math_prediction = svm_math.predict(user_df)
        reading_prediction = svm_reading.predict(user_df)
        writing_prediction = svm_writing.predict(user_df)

    return render_template("index.html", math_prediction=math_prediction, reading_prediction=reading_prediction, writing_prediction=writing_prediction)

if __name__ == "__main__":
    app.run(debug=True)
