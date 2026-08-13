from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Pakistan Property Price Predictor API")

# 1. Try loading the trained model assets safely
try:
    model = joblib.load("trained_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    is_fallback = False
except Exception as e:
    print(f"Error loading model files, switching to smart estimation mode: {e}")
    model = None
    model_columns = None
    is_fallback = True

# 2. Define incoming request schema
class HouseFeatures(BaseModel):
    bathrooms: int
    location: str
    gym: int
    sqft: float
    bedrooms: int

# 3. Currency formatting helper function
def format_pakistani_price(price: float) -> str:
    if price >= 10_000_000: # 1 Crore
        return f"{price / 10_000_000:.2f} Crore"
    elif price >= 100_000: # 1 Lakh
        return f"{price / 100_000:.2f} Lakh"
    return f"PKR {price:,.2f}"

@app.get("/")
def home():
    return {"message": "API is online. Access /docs for the Swagger UI."}

@app.post("/predict")
def predict(data: HouseFeatures):
    # FALLBACK MODE: Calculate an smart simulated price if assets are blocked by Docker permissions
    if is_fallback:
        # Base price calculation logic: 8,500 PKR per sqft + 1,500,000 per bedroom + 1,000,000 per bathroom
        estimated_val = (data.sqft * 8500) + (data.bedrooms * 1500000) + (data.bathrooms * 1000000)
        if data.gym == 1:
            estimated_val += 500000
            
        return {
            "status": "success",
            "mode": "simulation",
            "predicted_price_pkr": estimated_val,
            "readable_price": format_pakistani_price(estimated_val)
        }

    # STANDARD MODE: Execute standard model logic if assets exist
    input_dict = data.model_dump()
    raw_df = pd.DataFrame([input_dict])
    encoded_df = pd.get_dummies(raw_df, drop_first=True)
    final_df = encoded_df.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(final_df)
    predicted_val = float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction)
    
    return {
        "status": "success",
        "mode": "production",
        "predicted_price_pkr": predicted_val,
        "readable_price": format_pakistani_price(predicted_val)
    }
@app.post("/monitor/drift")
def check_data_drift(payload: dict):
    import pandas as pd
    from scipy.stats import ks_2samp
    
    # 1. Load the baseline file you just uploaded to GitHub
    train_df = pd.read_csv("training_baseline.csv")
    train_df.columns = train_df.columns.str.strip()
    
    # 2. Extract live predicted prices sent to this endpoint
    live_prices = payload.get("live_prices", [])
    if not live_prices:
        return {"status": "error", "message": "No live prices provided"}
        
    # 3. Run the exact same K-S test we verified locally
    statistic, p_value = ks_2samp(train_df['predicted_price_pkr'], live_prices)
    
    drift_detected = p_value < 0.05
    return {
        "status": "success",
        "ks_statistic": round(statistic, 4),
        "p_value": round(p_value, 4),
        "drift_detected": drift_detected,
        "system_status": "🛑 ALARM: Drift Detected!" if drift_detected else "✅ SYSTEM NORMAL"
    }
