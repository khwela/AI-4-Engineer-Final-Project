import unittest
import pandas as pd
from simulate_data import simulate_health_data, inject_anomalies
from detect_anomalies import detect_anomalies

class TestDetectAnomalies(unittest.TestCase):

    def test_detect_anomalies(self):
        df = simulate_health_data(100)
        df_with_anomalies = inject_anomalies(df.copy(), num=5)

        df_result, model = detect_anomalies(df_with_anomalies.copy())

        self.assertIn('anomaly_score', df_result.columns)
        self.assertIn('label', df_result.columns)

        # Check that the model identifies some anomalies
        self.assertTrue((df_result['label'] == 'Anomaly').any())

        # Check that the original data is not modified
        self.assertNotIn('anomaly_score', df_with_anomalies.columns)

if __name__ == '__main__':
    unittest.main()
