import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables or set defaults
MODEL_PATH = os.getenv('MODEL_PATH', 'models/random_forest_model.joblib')  # Modifiez le chemin vers votre modèle Random Forest

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="An API to predict customer churn based on customer data."
)

# Load the saved pipeline
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}.")
    raise
except Exception as e:
    logger.error(f"Error loading the model: {e}")
    raise


class CustomerFeatures(BaseModel):
    CreditScore: int = Field(..., gt=0, description="Credit score of the customer")
    Geography: str = Field(..., description="Geography (country) of the customer")
    Gender: str = Field(..., description="Gender of the customer (male or female)")
    Age: int = Field(..., gt=0, description="Age of the customer")
    Tenure: int = Field(..., ge=0, description="Number of years the customer has been with the bank")
    Balance: float = Field(..., ge=0, description="Account balance")
    NumOfProducts: int = Field(..., ge=1, description="Number of products the customer has")
    HasCrCard: int = Field(..., ge=0, le=1, description="Whether the customer has a credit card (1 = Yes, 0 = No)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Whether the customer is an active member (1 = Yes, 0 = No)")
    EstimatedSalary: float = Field(..., ge=0, description="Estimated salary of the customer")

@app.get("/", summary="Root Endpoint", description="Welcome to the Customer Churn Prediction API.")
def read_root():
    return {"message": "Welcome to the Customer Churn Prediction API. Use /predict to get predictions."}

@app.post("/predict", summary="Predict Customer Churn", description="Predict whether a customer will churn based on their data.")
async def predict_churn(features: CustomerFeatures):
    try:
        # Convert the input data to a DataFrame
        input_data = pd.DataFrame([features.dict().values()], columns=features.dict().keys())
        logger.info(f"Received input data: {input_data.to_dict(orient='records')}")
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        logger.info(f"Prediction: {prediction}")
        
        # Return the prediction result
        return {"prediction": "Churned" if prediction == 1 else "Not Churned"}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/health", summary="Health Check", description="Check if the API is running.")
def health_check():
    return {"status": "API is running"}
