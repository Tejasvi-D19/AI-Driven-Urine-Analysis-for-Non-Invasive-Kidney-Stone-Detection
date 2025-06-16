from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import Dict
import os

# Initialize FastAPI app
app = FastAPI(title="Kidney Stone Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input model
class PredictionInput(BaseModel):
    gravity: float
    ph: float
    osmo: float
    cond: float
    urea: float
    calc: float

# Load the model and scaler
MODEL_PATH = r"C:\Users\Teju\OneDrive\dsu\6th SEM\Minor project (MP)\Models\best_random_forest_model.pkl"
SCALER_PATH = r"C:\Users\Teju\OneDrive\dsu\6th SEM\Minor project (MP)\Models\scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    raise

def engineer_features(data: Dict) -> pd.DataFrame:
    """
    Perform feature engineering on input data
    """
    df = pd.DataFrame([data])
    
    # Interaction Terms
    df['urea_calc_squared'] = df['urea'] * (df['calc'] ** 2)
    df['osmo_urea_interaction'] = df['osmo'] * df['urea']
    df['cond_calc_ratio'] = df['cond'] / df['calc']
    
    # Clinical pH Binning
    df['ph_category'] = pd.cut(
        df['ph'],
        bins=[-np.inf, 5.5, 6.5, np.inf],
        labels=['acidic', 'neutral', 'alkaline']
    )
    ph_dummies = pd.get_dummies(df['ph_category'], prefix='ph_category')
    if 'ph_category_neutral' not in ph_dummies.columns:
        ph_dummies['ph_category_neutral'] = 0
    if 'ph_category_alkaline' not in ph_dummies.columns:
        ph_dummies['ph_category_alkaline'] = 0
    df = pd.concat([df, ph_dummies[['ph_category_neutral', 'ph_category_alkaline']]], axis=1)
    
    # Concentration Indicators
    df['total_solid_score'] = (df['osmo'] + df['gravity'] + df['cond']) / 3
    df['calc_osmo_ratio'] = df['calc'] / df['osmo']
    
    # Polynomial Features
    df['calc_squared'] = df['calc'] ** 2
    df['urea_sqrt'] = np.sqrt(df['urea'])
    
    # Reorder columns
    columns_order = [
        'gravity', 'ph', 'osmo', 'cond', 'urea', 'calc',
        'urea_calc_squared', 'osmo_urea_interaction', 'cond_calc_ratio',
        'ph_category_neutral', 'ph_category_alkaline',
        'total_solid_score', 'calc_osmo_ratio', 'calc_squared', 'urea_sqrt'
    ]
    
    return df[columns_order]

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Convert input to dictionary
        data_dict = input_data.dict()
        
        # Engineer features
        df = engineer_features(data_dict)
        
        # Columns to scale
        scale_columns = [
            'gravity', 'ph', 'osmo', 'cond', 'urea', 'calc',
            'urea_calc_squared', 'osmo_urea_interaction', 'cond_calc_ratio',
            'total_solid_score', 'calc_osmo_ratio', 'calc_squared', 'urea_sqrt'
        ]
        
        # Scale features
        df[scale_columns] = scaler.transform(df[scale_columns])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability[1]),
            "message": "Kidney Stone Detected" if prediction == 1 else "No Kidney Stone Detected"
        }
        
    except ZeroDivisionError:
        raise HTTPException(status_code=400, detail="Division by zero error in feature engineering. Please check your input values.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 