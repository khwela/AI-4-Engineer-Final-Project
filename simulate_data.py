import pandas as pd
import numpy as np

def simulate_health_data(minutes=100):
    """
    Simulates wearable health data for a specified number of minutes.
    """
    start_time = pd.to_datetime('2023-10-01 00:00')
    timestamps = pd.to_datetime([start_time + pd.Timedelta(minutes=i) for i in range(minutes)])

    heart_rate = np.random.randint(60, 101, size=minutes)
    blood_oxygen = np.random.randint(90, 101, size=minutes)
    activities = ['low', 'moderate', 'high']
    activity_level = np.random.choice(activities, size=minutes, p=[0.6, 0.3, 0.1])

    df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': heart_rate,
        'blood_oxygen': blood_oxygen,
        'activity_level': activity_level
    })

    return df

def inject_anomalies(df, num=5):
    """
    Injects anomalies into the health data DataFrame.
    """
    anomaly_indices = np.random.choice(df.index, size=num, replace=False)

    for i in anomaly_indices:
        df.loc[i, 'heart_rate'] = np.random.randint(121, 150)
        df.loc[i, 'blood_oxygen'] = np.random.randint(80, 88)

    return df

if __name__ == '__main__':
    simulated_data = simulate_health_data(100)
    final_data = inject_anomalies(simulated_data)

    print("Simulated Data with Anomalies:")
    print(final_data.head())
    print("\n...")
    print(final_data.tail())

    # Describe the data to show the effect of anomalies
    print("\nData Description:")
    print(final_data.describe())
