# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Loading the dataset
df = pd.read_csv(r'C:\Users\akw97\Desktop\Parami Courses\Advanced_Machine_Learning\flask_apps\Loan_Flask\loan_data.csv')

# Dropping the 'loan_status' column
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Initialize LabelEncoder and StandardScaler
le = LabelEncoder()
sc = StandardScaler()

# Label Encoding for categorical columns
categorical_cols = X.select_dtypes(include='object').columns
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Standardizing numerical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[numerical_cols] = sc.fit_transform(X[numerical_cols])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)

# Using KNN Classifier to train the model
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance', p=2)
knn.fit(X_train, y_train)

# Exporting the model
joblib.dump(knn, 'loan_pred_model.joblib')
print('Model has been successfully exported')

# Exporting the Label Encoder
joblib.dump(le, 'label_encoder.joblib')
print('Label Encoder has been successfully exported')

# Exporting the Standard Scaler
joblib.dump(sc, 'standard_scaler.joblib')
print('Standard Scaler has been successfully exported')