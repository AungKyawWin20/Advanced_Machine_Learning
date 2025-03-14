import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Load the dataset
def load_data(file):
    df = pd.read_csv(file, sep='\t', header=None, names=['target', 'text'])
    return df

# Split the data into training and testing
def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Vectorize the text
def vectorize_text(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, vectorizer

# Train the model
def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# Main script
if __name__ == "__main__":
    df = load_data(r'C:\Users\akw97\Desktop\Parami Courses\Advanced Machine Learning\Advanced_Machine_Learning\Lectures\SMSSpamCollection')
    
    # Map target values
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, vectorizer = vectorize_text(X_train, X_test)
    model = train_model(X_train, y_train)
    
    # Save the model and vectorizer
    #joblib.dump(model, 'spam_model.pkl')
    #joblib.dump(vectorizer, 'vectorizer.pkl')
    
    print("Model and vectorizer saved!")