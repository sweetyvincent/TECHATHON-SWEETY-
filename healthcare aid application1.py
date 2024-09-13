Python 3.9.6 (tags/v3.9.6:db3ff76, Jun 28 2021, 15:26:21) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import statistics

# Load the dataset
data = pd.read_csv("dataset/Disease Symptom Description Dataset.csv").dropna(axis=1)

# Create a dictionary to map symptoms to indices
data_dict = {}
data_dict["symptom_index"] = {symptom: index for index, symptom in enumerate(data.columns[1:])}

# Encode the target value into numerical value using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
data_dict["predictions_classes"] = {index: prognosis for index, prognosis in enumerate(encoder.classes_)}

# Split the data into training and testing sets
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Train the models
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X_train, y_train)
final_nb_model.fit(X_train, y_train)
final_rf_model.fit(X_train, y_train)

# Make predictions on the test data
svm_preds = final_svm_model.predict(X_test)
nb_preds = final_nb_model.predict(X_test)
rf_preds = final_rf_model.predict(X_test)

# Evaluate the models using accuracy score and confusion matrix
print("SVM Accuracy:", accuracy_score(y_test, svm_preds))
print("NB Accuracy:", accuracy_score(y_test, nb_preds))
print("RF Accuracy:", accuracy_score(y_test, rf_preds))

cf_matrix = confusion_matrix(y_test, svm_preds)
print("SVM Confusion Matrix:")
print(cf_matrix)

cf_matrix = confusion_matrix(y_test, nb_preds)
print("NB Confusion Matrix:")
print(cf_matrix)

cf_matrix = confusion_matrix(y_test, rf_preds)
print("RF Confusion Matrix:")
print(cf_matrix)

# Function to predict disease from symptoms
def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
    input_data = np.array(input_data).reshape(1,-1)
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction":final_prediction
    }
    return predictions