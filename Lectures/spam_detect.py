import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import joblib

#Load the datasert
def load_data(file):
    df = pd.read_csv(file, sep='\t', header=None,names=['target', 'text'])
    return df
#Split the data into training and testing
def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

#Vectorize the text
def vectorize_text(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    y_train = y_train.map({'ham': 0, 'spam': 1})
    y_test = y_test.map({'ham': 0, 'spam': 1})
    return X_train, X_test, vectorizer

#Train the model
def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


