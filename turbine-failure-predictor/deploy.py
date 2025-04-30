# Turbine Failure Prediction Deployment
import joblib
import pandas as pd

def load_model():
    return joblib.load('turbine_failure_predictor.pkl')

def predict(input_csv):
    assets = load_model()
    data = pd.read_csv(input_csv)
    return assets['model'].predict(data[assets['features']])

if __name__ == '__main__':
    import sys
    predictions = predict(sys.argv[1])
    pd.Series(predictions).to_csv('predictions.csv', index=False)
