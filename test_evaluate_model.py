import unittest
import pandas as pd
from simulate_data import simulate_health_data, inject_anomalies
from detect_anomalies import detect_anomalies
from evaluate_model import evaluate_model

class TestEvaluateModel(unittest.TestCase):

    def test_evaluate_model(self):
        # Generate and label data
        simulated_data = simulate_health_data(200)
        data_with_anomalies = inject_anomalies(simulated_data.copy(), num=10)
        labeled_df, _ = detect_anomalies(data_with_anomalies.copy())

        # Evaluate the model
        report = evaluate_model(labeled_df)

        # Check that the report is a dictionary
        self.assertIsInstance(report, dict)

        # Check for expected keys in the report
        self.assertIn('0', report) # Normal
        self.assertIn('1', report) # Anomaly
        self.assertIn('accuracy', report)
        self.assertIn('macro avg', report)
        self.assertIn('weighted avg', report)

if __name__ == '__main__':
    unittest.main()
