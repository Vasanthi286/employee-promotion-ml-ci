import unittest
import joblib
import pandas as pd


class TestEmployeePromotionAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load("employee_promotion_model.pkl")

    def test_model_can_be_loaded(self):
        # Restored correct test
        self.assertIsNotNone(self.model)

    def test_prediction_can_be_generated(self):

        sample_data = pd.DataFrame([{
            "Age": 30,
            "Department": "R&D",
            "Education_Level": "Bachelor's",
            "Years_at_Company": 5,
            "Years_in_Current_Role": 2,
            "Job_Level": "Mid",
            "Performance_Score": 4,
            "Training_Hours": 20,
            "Projects_Completed": 5,
            "Certifications": 2,
            "Salary": 60000,
            "Job_Satisfaction": 4,
            "Work_Life_Balance": 4,
            "Manager_Rating": 4,
            "Overtime": "No",
            "Previous_Promotions": 1,
            "Absence_Days": 5,
            "Remote_Work": "Yes",
            "Monthly_Hours": 160
        }])

        prediction = self.model.predict(sample_data)

        self.assertEqual(len(prediction), 1)


if __name__ == "__main__":
    unittest.main()
