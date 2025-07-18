import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest

from simulate_data import simulate_health_data, inject_anomalies
from detect_anomalies import detect_anomalies

def evaluate_model(df):
    """
    Evaluates the Isolation Forest model's performance.
    """
    # Prepare data
    df['label_binary'] = df['label'].apply(lambda x: 1 if x == 'Anomaly' else 0)
    features = df[['heart_rate', 'blood_oxygen']]
    labels = df['label_binary']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    # Train model
    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(X_train)

    # Predict on test data
    y_pred_scores = model.predict(X_test)
    y_pred = [1 if x == -1 else 0 for x in y_pred_scores]

    # Evaluate
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

    return classification_report(y_test, y_pred, output_dict=True)

if __name__ == '__main__':
    # Generate and label data
    simulated_data = simulate_health_data(200) # Larger dataset for more meaningful evaluation
    data_with_anomalies = inject_anomalies(simulated_data.copy(), num=10)
    labeled_df, _ = detect_anomalies(data_with_anomalies.copy())

    # Evaluate the model
    evaluate_model(labeled_df)
