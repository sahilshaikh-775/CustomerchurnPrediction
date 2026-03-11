import pandas as pd
from src.customerchurn.pipelines.prediction_pipeline import PredictionPipeline


def test_predict_returns_expected_keys():
    pipe = PredictionPipeline()

    row = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }

    out = pipe.predict(pd.DataFrame([row]))

    assert "churn_probability" in out
    assert "churn_prediction" in out
    assert 0.0 <= float(out["churn_probability"]) <= 1.0
    assert int(out["churn_prediction"]) in (0, 1)
