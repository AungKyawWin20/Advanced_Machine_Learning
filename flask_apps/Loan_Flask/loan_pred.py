#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import joblib

#Loading the dataset       
df = pd.read_csv('flask_apps\KNN Flask(Loan_Default)\loan_data.csv')

#Dropping the 'loan_status' column
X = df.drop('loan_status', axis=1)

y = df['loan_status']

#Label Encoding the categorical columns and standardizing the numerical columns
le = LabelEncoder()
sc = StandardScaler()

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])
    else:
        X[col] = sc.fit_transform(X[[col]])
        
#Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)

#Using KNN Classifier to train the model
#KNN Classifier with 5 neighbors, manhattan distance metric, distance weights and p=2
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights = 'distance', p=2)

knn.fit(X_train, y_train)

#Exporting the model
joblib.dump(knn, 'loan_pred_model.joblib')
print('Model has been successfully exported')

#Exporting the Label Encoder
joblib.dump(le, 'label_encoder.joblib')
print('Label Encoder has been successfully exported')

#Exporting the Standard Scaler
joblib.dump(sc, 'standard_scaler.joblib')
print('Standard Scaler has been successfully exported')