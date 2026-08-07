from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Pakistan Property Price Predictor API")

# 1. Load the trained model assets
try:
    model = joblib.load("trained_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None
    model_columns = None

# 2. Define incoming request schema
class HouseFeatures(BaseModel):
    bathrooms: int
    location: str
    gym: int
    sqft: float
    bedrooms: int

# 3. Currency formatting helper function
def format_pakistani_price(price: float) -> str:
    if price >= 10_000_000:       # 1 Crore
        return f"{price / 10_000_000:.2f} Crore"
    elif price >= 100_000:        # 1 Lakh
        return f"{price / 100_000:.2f} Lakh"
    return f"PKR {price:,.2f}"

@app.get("/")
def home():
    return {"message": "API is online. Access /docs for the Swagger UI."}

@app.post("/predict")
def predict(data: HouseFeatures):
    if model is None or model_columns is None:
        return {"error": "Model files are missing or not loaded correctly."}
        
    # Convert incoming API request to a single-row DataFrame
    input_dict = data.model_dump()
    raw_df = pd.DataFrame([input_dict])
    
    # Replicate pd.get_dummies processing dynamically
    encoded_df = pd.get_dummies(raw_df, drop_first=True)
    
    # Align rows with the training columns layout
    final_df = encoded_df.reindex(columns=model_columns, fill_value=0)
    
    # Execute prediction
    prediction = model.predict(final_df)
    predicted_val = float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction)
    
    return {
        "status": "success",
        "predicted_price_pkr": predicted_val,
        "readable_price": format_pakistani_price(predicted_val)
    }
