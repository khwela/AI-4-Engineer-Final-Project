import pandas as pd
from sklearn.ensemble import IsolationForest
from simulate_data import simulate_health_data, inject_anomalies

def detect_anomalies(df):
    """
    Detects anomalies in the health data using an Isolation Forest model.
    """
    features = df[['heart_rate', 'blood_oxygen']]

    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(features)

    df['anomaly_score'] = model.predict(features)
    df['label'] = df['anomaly_score'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

    return df, model

if __name__ == '__main__':
    # Generate data
    simulated_data = simulate_health_data(100)
    data_with_anomalies = inject_anomalies(simulated_data.copy(), num=5)

    # Detect anomalies
    final_df, trained_model = detect_anomalies(data_with_anomalies.copy())

    print("Data with Anomaly Detection:")
    print(final_df.head())
    print("\n...")
    print(final_df.tail())

    print("\nAnomalies Detected:")
    print(final_df[final_df['label'] == 'Anomaly'])

    print("\nModel Info:")
    print(trained_model)
