import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(
            os.path.exists("employee_promotion_raw.xlsx")
        )

    def test_model_created(self):
        self.assertTrue(
            os.path.exists("employee_promotion_model.pkl")
        )

    def test_metrics_created(self):
        self.assertTrue(
            os.path.exists("metrics.json")
        )

    def test_dataset_size(self):
        data = pd.read_excel(
            "employee_promotion_raw.xlsx"
        )

        self.assertEqual(len(data), 3000)

    def test_accuracy_is_valid(self):

        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):

        model = joblib.load(
            "employee_promotion_model.pkl"
        )

        sample = pd.DataFrame([{
            "Age": 30,
            "Department": "R&D",
            "Education_Level": "Master's",
            "Years_at_Company": 8,
            "Years_in_Current_Role": 2,
            "Job_Level": "Mid",
            "Performance_Score": 5,
            "Training_Hours": 80,
            "Projects_Completed": 15,
            "Certifications": 4,
            "Salary": 120000,
            "Job_Satisfaction": 5,
            "Work_Life_Balance": 5,
            "Manager_Rating": 5,
            "Overtime": "No",
            "Previous_Promotions": 2,
            "Absence_Days": 3,
            "Remote_Work": "Yes",
            "Monthly_Hours": 170
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(
            int(prediction),
            [0, 1]
        )

    def test_high_promotion_candidate(self):

        model = joblib.load(
            "employee_promotion_model.pkl"
        )

        sample = pd.DataFrame([{
            "Age": 30,
            "Department": "R&D",
            "Education_Level": "Master's",
            "Years_at_Company": 8,
            "Years_in_Current_Role": 2,
            "Job_Level": "Mid",
            "Performance_Score": 5,
            "Training_Hours": 80,
            "Projects_Completed": 15,
            "Certifications": 4,
            "Salary": 120000,
            "Job_Satisfaction": 5,
            "Work_Life_Balance": 5,
            "Manager_Rating": 5,
            "Overtime": "No",
            "Previous_Promotions": 2,
            "Absence_Days": 3,
            "Remote_Work": "Yes",
            "Monthly_Hours": 170
        }])

        prediction = model.predict(sample)[0]

        self.assertEqual(
            int(prediction),
            1
        )

    def test_low_promotion_candidate(self):

        model = joblib.load(
            "employee_promotion_model.pkl"
        )

        sample = pd.DataFrame([{
            "Age": 30,
            "Department": "R&D",
            "Education_Level": "Bachelor's",
            "Years_at_Company": 1,
            "Years_in_Current_Role": 1,
            "Job_Level": "Junior",
            "Performance_Score": 1,
            "Training_Hours": 5,
            "Projects_Completed": 2,
            "Certifications": 0,
            "Salary": 40000,
            "Job_Satisfaction": 1,
            "Work_Life_Balance": 1,
            "Manager_Rating": 1,
            "Overtime": "Yes",
            "Previous_Promotions": 0,
            "Absence_Days": 25,
            "Remote_Work": "No",
            "Monthly_Hours": 220
        }])

        prediction = model.predict(sample)[0]

        self.assertEqual(
            int(prediction),
            0
        )


if __name__ == "__main__":
    unittest.main()
