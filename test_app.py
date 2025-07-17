import unittest
import json
from app import app

class TestApi(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_normal(self):
        payload = {"heart_rate": 85, "blood_oxygen": 96}
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data['status'], 'Normal')

    def test_predict_anomaly(self):
        payload = {"heart_rate": 130, "blood_oxygen": 85}
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data['status'], 'Anomaly')

    def test_missing_data(self):
        payload = {"heart_rate": 85}
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('error', data)

    def test_invalid_data(self):
        payload = {"heart_rate": "high", "blood_oxygen": 96}
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()
