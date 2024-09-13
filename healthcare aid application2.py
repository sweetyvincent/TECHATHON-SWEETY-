import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("dataset/Healthcare Insurance.csv")

# Encode categorical variables using LabelEncoder
encoder = LabelEncoder()
data["sex"] = encoder.fit_transform(data["sex"])
data["smoker"] = encoder.fit_transform(data["smoker"])
data["region"] = encoder.fit_transform(data["region"])

# Split the data into training and testing sets
X = data.drop(["charges"], axis=1)
y = data["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Train the model
final_model = LinearRegression()
final_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = final_model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# Function to estimate insurance co-pay/premium
def estimateInsuranceCoPay(bmi, smoker, region):
    input_data = [bmi, smoker, region]
    input_data = np.array(input_data).reshape(1,-1)
    prediction = final_model.predict(input_data)[0]
    return prediction
