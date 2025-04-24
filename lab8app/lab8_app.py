from fastapi import FastAPI, HTTPException, Request
import mlflow
import numpy as np
from pydantic import BaseModel
from typing import List
import uvicorn
import pandas as pd
import mlflow.pyfunc


# Initialize FastAPI app
app = FastAPI(title="Car Price Prediction API")

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("car_prediction_model")
logged_model = 'mlflow-artifacts:/267996663564729696/d38a43b6326e47909a12d99d1d08d77c/artifacts/car_price_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

@app.post("/predict")
async def predict(request: Request):
    payload = await request.json()
    input_data = payload.get("data")
    print(input_data)
    columns = [
    'Spare key',
    'KM driven',
    'Ownership',
    'Imperfections',
    'Repainted Parts',
    'Transmission_Manual',
    'Fuel type_Diesel',
    'Fuel type_Petrol'
    ]

    input_df = pd.DataFrame(input_data, columns=columns)
    print(input_df)

    prediction = loaded_model.predict(input_df)
    return {"prediction": prediction.tolist()}

