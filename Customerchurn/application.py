from flask import Flask, request,jsonify,render_template
from src.customerchurn.pipelines.prediction_pipeline import PredictionPipeline


application = Flask(__name__)

predictor = PredictionPipeline()

OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No internet service", "No", "Yes"],
    "OnlineBackup": ["No internet service", "No", "Yes"],
    "DeviceProtection": ["No internet service", "No", "Yes"],
    "TechSupport": ["No internet service", "No", "Yes"],
    "StreamingTV": ["No internet service", "No", "Yes"],
    "StreamingMovies": ["No internet service", "No", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


@application.get('/health')
def health():
    return jsonify({'status':"ok"})

@application.route('/', methods=["GET","POST"])
def home():
    result = None
    error = None

    form_data = {
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": "1",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": "89.10",
        "TotalCharges": "89.10",
    }

    if request.method=="POST":
        try:
            form_data = {k:request.form.get(k,"").strip() for k in form_data.keys()}

            payload = {
                "gender": form_data["gender"],
                "SeniorCitizen": int(form_data["SeniorCitizen"]),
                "Partner": form_data["Partner"],
                "Dependents": form_data["Dependents"],
                "tenure": int(form_data["tenure"]),
                "PhoneService": form_data["PhoneService"],
                "MultipleLines": form_data["MultipleLines"],
                "InternetService": form_data["InternetService"],
                "OnlineSecurity": form_data["OnlineSecurity"],
                "OnlineBackup": form_data["OnlineBackup"],
                "DeviceProtection": form_data["DeviceProtection"],
                "TechSupport": form_data["TechSupport"],
                "StreamingTV": form_data["StreamingTV"],
                "StreamingMovies": form_data["StreamingMovies"],
                "Contract": form_data["Contract"],
                "PaperlessBilling": form_data["PaperlessBilling"],
                "PaymentMethod": form_data["PaymentMethod"],
                "MonthlyCharges": float(form_data["MonthlyCharges"]),
                "TotalCharges": float(form_data["TotalCharges"]),
            }

            result = predictor.predict(payload)

        except ValueError as ve:
            error = f"Invalid number format: {ve}"
        except Exception as e:
            error = str(e)     
    return render_template("index.html",options=OPTIONS,form_data=form_data,result=result,error=error)

@application.post("/predict")
def predic_api():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error":"Invalid JSON"}), 400
    return jsonify(predictor.predict(payload))

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000, debug=True)
    