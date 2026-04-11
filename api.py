from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Hybrid AML Prediction Service")

# Configuration
PROCESSED_DIR = r'e:\MASTERSProject\AMLProject\processed_data'

# In a production environment, we'd load a pre-trained model from disk (e.g., model.joblib)
# For the demo, we ensure processed data is loaded
def load_model():
    # If model doesn't exist, we train it on load (simulated)
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train.csv')).values.ravel()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = load_model()

class Transaction(BaseModel):
    amount: float
    source_degree: float
    target_degree: float
    source_betweenness: float = 0.0
    target_betweenness: float = 0.0
    community_id: int = 0
    tx_count_step: int = 1
    is_small_amount: int = 0
    hist_tx_count: int = 0
    amount_vs_hist_mean: float = 1.0

@app.get("/")
def read_root():
    return {"status": "online", "service": "Hybrid AML System (MoMTSim)"}

@app.post("/predict")
def predict_aml(tx: Transaction):
    try:
        # Construct input for model
        input_data = np.array([[
            tx.amount, tx.source_degree, tx.target_degree, 
            tx.source_betweenness, tx.target_betweenness, 
            tx.community_id, tx.tx_count_step, tx.is_small_amount,
            tx.hist_tx_count, tx.amount_vs_hist_mean
        ]])

        
        # Inference
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return {
            "status": "SUSPICIOUS" if prediction == 1 else "NORMAL",
            "risk_score": float(probability),
            "flags": ["High SME/SNA Risk"] if prediction == 1 else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
