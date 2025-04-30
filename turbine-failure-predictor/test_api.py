# Add this to deploy.py for API usage
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
async def predict_api(data: dict):
    return {"prediction": bool(predict(pd.DataFrame([data]))[0])}