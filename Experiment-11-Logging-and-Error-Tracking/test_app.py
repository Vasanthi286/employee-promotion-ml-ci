
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure the Practical 11 folder is on Python's import path
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import app as app_module


class TestEmployeePromotionAPI(unittest.TestCase):

    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "Employee Promotion Prediction API is running",
            response.get_json()["message"]
        )

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "healthy")

    def test_invalid_prediction(self):
        response = self.client.post(
            "/predict",
            data=json.dumps({}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

    def test_unknown_route(self):
        response = self.client.get("/unknown")
        self.assertEqual(response.status_code, 404)

    @patch.object(app_module.model, "predict", return_value=[1])
    def test_prediction_success(self, mock_predict):
        response = self.client.post(
            "/predict",
            json={"Age": 30}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["promotion_prediction"], 1
        )
        mock_predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
