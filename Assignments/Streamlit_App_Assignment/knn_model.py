import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv(r'Advanced_Machine_Learning\Assignments\Streamlit_App_Assignment\IRIS.csv ') #Importing the categorical dataset

df.dropna(inplace=True)

X = df.drop(columns=['species'])
y = df['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#Generate a model

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn,f)