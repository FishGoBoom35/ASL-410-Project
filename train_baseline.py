import pandas as pd

# Load the cleaned ASL dataset
df = pd.read_csv("data/cleaned_dataset.csv")

# Display dataset information
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.iloc[:, :8].head())

print("\nNumber of samples per ASL word:")
print(df["label"].value_counts())

# Separate input features from labels
X = df.drop(columns=["label", "Unnamed: 0"], errors="ignore")
y = df["label"]

print("\nTotal samples:", len(X))
print("Number of features:", X.shape[1])
print("Number of ASL classes:", y.nunique())

from sklearn.model_selection import train_test_split

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Create Logistic Regression model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

# Train using the 360 training samples
model.fit(X_train, y_train)

# Predict the 90 testing samples
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Logistic Regression Results ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Predict the test samples
rf_pred = rf_model.predict(X_test)

# Evaluate performance
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n--- Random Forest Results ---")
print(f"Accuracy: {rf_accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))


from sklearn.svm import SVC

# Create SVM model
svm_model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0)
)

# Train the model
svm_model.fit(X_train, y_train)

# Predict test samples
svm_pred = svm_model.predict(X_test)

# Evaluate performance
svm_accuracy = accuracy_score(y_test, svm_pred)

print("\n--- SVM Results ---")
print(f"Accuracy: {svm_accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, svm_pred))


from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# XGBoost requires numerical class labels
encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Create XGBoost model
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="mlogloss",
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train_encoded)

# Predict the testing samples
xgb_pred = xgb_model.predict(X_test)

# Evaluate accuracy
xgb_accuracy = accuracy_score(y_test_encoded, xgb_pred)

print("\n--- XGBoost Results ---")
print(f"Accuracy: {xgb_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test_encoded,
    xgb_pred,
    target_names=encoder.classes_
))


from sklearn.neighbors import KNeighborsClassifier

# Create KNN model
knn_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

# Train model
knn_model.fit(X_train, y_train)

# Predict test samples
knn_pred = knn_model.predict(X_test)

# Evaluate results
knn_accuracy = accuracy_score(y_test, knn_pred)

print("\n--- KNN Results ---")
print(f"Accuracy: {knn_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, knn_pred))
