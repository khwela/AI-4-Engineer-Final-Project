import unittest
import pandas as pd
from simulate_data import simulate_health_data, inject_anomalies

class TestSimulateData(unittest.TestCase):

    def test_simulate_health_data(self):
        df = simulate_health_data(100)
        self.assertEqual(len(df), 100)
        self.assertTrue(all(df['heart_rate'].between(60, 100)))
        self.assertTrue(all(df['blood_oxygen'].between(90, 100)))
        self.assertTrue(all(df['activity_level'].isin(['low', 'moderate', 'high'])))

    def test_inject_anomalies(self):
        df = simulate_health_data(100)
        df_anomalies = inject_anomalies(df.copy(), num=5)

        # Count anomalies
        anomalies = df_anomalies[(df_anomalies['heart_rate'] > 120) | (df_anomalies['blood_oxygen'] < 88)]
        self.assertEqual(len(anomalies), 5)

        # Check that original data is not modified
        self.assertTrue(all(df['heart_rate'].between(60, 100)))
        self.assertTrue(all(df['blood_oxygen'].between(90, 100)))

if __name__ == '__main__':
    unittest.main()
