
import json
import unittest
from unittest.mock import patch

import app as app_module


class TestEmployeePromotionAPI(unittest.TestCase):

    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

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
        self.assertEqual(response.get_json()["promotion_prediction"], 1)


if __name__ == "__main__":
    unittest.main()

