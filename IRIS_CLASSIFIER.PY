Machine Learning Model Classifier
----------------------------------
This script loads the classic Iris dataset, splits it into training and test sets,
trains a Random Forest Classifier, evaluates its accuracy, and makes predictions.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def run_pipeline():
    # 1. Load Dataset
    print("Loading Iris dataset...")
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names

    print(f"Features: {feature_names}")
    print(f"Target Classes: {list(target_names)}\n")

    # 2. Split Data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train Random Forest Model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 5. Make Sample Prediction
    sample_flower = [[5.1, 3.5, 1.4, 0.2]]  # Setosa measurements
    prediction = model.predict(sample_flower)
    print(f"Prediction for sample {sample_flower}: {target_names[prediction[0]]}")


if _name_ == "_main_":
    run_pipeline()