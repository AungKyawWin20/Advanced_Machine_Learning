import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
file_path = "loan_data.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
categorical_cols = ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]
df_encoded = df.copy()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Define features and target variable
X = df_encoded.drop(columns=["loan_status"])
y = df_encoded["loan_status"]

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Saving the model and preprocessing objects
joblib.dump(knn, "knn_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(label_encoders, "label_encoders.joblib")

print("KNN model saved successfully!")
