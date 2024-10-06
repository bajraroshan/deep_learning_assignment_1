import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading Dataset 
data = pd.read_csv('diabetes.csv')

# Data Preprocessing
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_to_replace] = data[columns_to_replace].replace(0, np.nan)


imputer = SimpleImputer(strategy='median')
data[columns_to_replace] = imputer.fit_transform(data[columns_to_replace])

# Feature Scaling
scaler = StandardScaler()
features = data.columns.drop('Outcome')
data[features] = scaler.fit_transform(data[features])

# Step 2: Splitting the Dataset

Y = data.drop('Outcome', axis=1)
N = data['Outcome']

#Data Splitting
Y_train, Y_test, N_train, N_test = train_test_split(Y, N, test_size=0.4, random_state=300, stratify=N)


# Model Selection
# Initialize the Perceptron
model = Perceptron(max_iter=1000, random_state=42)  

# Experimental analysis and testing

# Train the model
model.fit(Y_train, N_train)

# Make predictions
N_pred = model.predict(Y_test)

# Evaluate the model
accuracy = accuracy_score(N_test, N_pred)
conf_matrix = confusion_matrix(N_test, N_pred)
class_report = classification_report(N_test, N_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)