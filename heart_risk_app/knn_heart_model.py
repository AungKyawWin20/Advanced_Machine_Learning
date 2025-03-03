import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('dataset.csv')
X = df.drop(columns=['target'])
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled,y)

#Generate a model

with open('knn_heart_model.pkl', 'wb') as f:
    pickle.dump(knn,f)
    
with open('knn_scaler.pkl', 'wb') as f:
    pickle.dump(scaler,f)
    
print("Done")
