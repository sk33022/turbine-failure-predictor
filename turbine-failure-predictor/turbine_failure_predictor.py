#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
df = pd.read_csv('t1.csv')
df.head(2)


# In[9]:


df.info()


# In[12]:


df.isnull().sum()


# In[13]:


import numpy as np

# Convert to datetime and sort
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')  # Matches our date format
df = df.set_index('Date/Time').sort_index()

# Efficiency feature
df['Efficiency'] = df['LV ActivePower (kW)'] / df['Theoretical_Power_Curve (KWh)']

# Rolling stats (6-hour window)
df['Power_6h_avg'] = df['LV ActivePower (kW)'].rolling('6h').mean()
df['Power_6h_std'] = df['LV ActivePower (kW)'].rolling('6h').std()

# Lag features (for 48h prediction)
for lag in [6, 24, 48]:  # 6h, 24h, 48h lags
    df[f'Power_lag_{lag}'] = df['LV ActivePower (kW)'].shift(lag)
df = df.dropna()


# In[14]:


# Failure = 1 if power < 50% theoretical AND wind speed > 3 m/s
df['Failure'] = np.where(
    (df['LV ActivePower (kW)'] < 0.5 * df['Theoretical_Power_Curve (KWh)']) & 
    (df['Wind Speed (m/s)'] > 3),
    1, 0
)

print("Failure distribution:\n", df['Failure'].value_counts())


# In[22]:



# 1. Create Features (with NaN/inf protection)
def create_features(df):
    # Safe efficiency calculation
    df['Efficiency'] = np.divide(
        df['LV ActivePower (kW)'],
        df['Theoretical_Power_Curve (KWh)'],
        out=np.zeros_like(df['LV ActivePower (kW)']),
        where=(df['Theoretical_Power_Curve (KWh)'] != 0)
    )
    
    # Rolling features (6h window)
    df['Power_6h_avg'] = df['LV ActivePower (kW)'].rolling('6h', min_periods=1).mean()
    df['Power_6h_std'] = df['LV ActivePower (kW)'].rolling('6h', min_periods=1).std()
    
    # Lag features
    for lag in [6, 24, 48]:  # 6h, 24h, 48h lags
        df[f'Power_lag_{lag}'] = df['LV ActivePower (kW)'].shift(lag)
    
    return df
df = create_features(df)


# In[23]:



# 2. Clean Data
def clean_data(df):
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Forward-fill remaining NaNs (for lags)
    df = df.ffill()
    
    # Drop any remaining NaNs (should be minimal)
    df = df.dropna()
    
    # Final sanity check
    assert not df.isna().any().any(), "NaNs still present!"
    assert not np.isinf(df.select_dtypes(include=np.number)).any().any(), "Infs still present!"
    
    return df

df = clean_data(df)

  


# In[24]:



# 3. Prepare Modeling Data
X = df[[
    'LV ActivePower (kW)', 'Wind Speed (m/s)', 'Efficiency',
    'Power_6h_avg', 'Power_6h_std', 'Power_lag_6', 'Power_lag_48'
]]
y = df['Failure']


# In[25]:



# 4. Train-Test Split (Time-series aware)
train_size = int(0.8 * len(df))  # 80% train, 20% test
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# 5. Train Model
model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_estimators=100
)
model.fit(X_train, y_train)


# In[26]:



# 6. Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
print("\nFeature Importances:")
print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))


# In[33]:


# CONTINUING AFTER FEATURE IMPORTANCE PRINTING

# 1. Get the most important features (top 5)
top_features = model.feature_names_in_[np.argsort(model.feature_importances_)[-5:]]
print("\nTop 5 Features:", top_features)

# 2. Create reduced feature set
X_train_reduced = X_train[top_features]
X_test_reduced = X_test[top_features]

# 3. Retrain model on best features
optimized_model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_estimators=150  # Slightly larger for better performance
)
optimized_model.fit(X_train_reduced, y_train)

# 4. Evaluate optimized model
y_pred_opt = optimized_model.predict(X_test_reduced)
print("\nOptimized Model Report:")
print(classification_report(y_test, y_pred_opt))

# 5. Threshold tuning (using reduced features)
y_probs = optimized_model.predict_proba(X_test_reduced)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Find threshold for 90% recall
target_recall = 0.9
optimal_idx = np.argmax(recall >= target_recall)
optimal_threshold = thresholds[optimal_idx]
print(f"\nOptimal Threshold for {target_recall:.0%} Recall: {optimal_threshold:.3f}")

# 6. Final predictions with threshold
final_predictions = (y_probs >= optimal_threshold).astype(int)
print("\nFinal Evaluation:")
print(classification_report(y_test, final_predictions))

# 7. Save everything needed for production
import joblib
production_assets = {
    'model': optimized_model,
    'features': top_features.tolist(),
    'threshold': optimal_threshold
}
joblib.dump(production_assets, 'turbine_failure_predictor.pkl')

# 8. Create prediction function template
def predict_failure(new_data, model_path='turbine_failure_predictor.pkl'):
    """Usage: df['Failure_Risk'] = predict_failure(new_data)"""
    assets = joblib.load(model_path)
    required_cols = assets['features']
    probs = assets['model'].predict_proba(new_data[required_cols])[:,1]
    return probs >= assets['threshold']

print("\n✅ Pipeline Complete - Ready for Production!")


# In[34]:


# Load and test
test_data = X_test.iloc[:5][top_features]  # First 5 rows
predictions = predict_failure(test_data)
print("Test Predictions:", predictions)


# In[38]:


# Test with NEW unseen data (not X_test)
new_samples = pd.DataFrame({
    'LV ActivePower (kW)': [350, 420, 380],
    'Wind Speed (m/s)': [5.2, 6.1, 5.8],
    'Efficiency': [0.82, 0.91, 0.45],  # Example: Last one should predict failure
    'Power_6h_avg': [360, 410, 370],
    'Power_6h_std': [12, 15, 50]  # High std might indicate issues
})[top_features]  # Ensure same feature order

print("New Data Predictions:")
print(predict_failure(new_samples))


# In[39]:


assets = joblib.load('turbine_failure_predictor.pkl')
print("Model requires:", assets['features'])


# In[41]:


def create_test_sample():
    return pd.DataFrame({
        'LV ActivePower (kW)': [float(input("Current Power (kW): "))],
        'Wind Speed (m/s)': [float(input("Wind Speed (m/s): "))],
        'Efficiency': [float(input("Efficiency (0-1): "))],
        'Power_6h_avg': [float(input("6h Avg Power (kW): "))],
        'Power_6h_std': [float(input("6h Power Std Dev: "))],
        'Power_lag_6': [float(input("Power 6h ago (kW): "))],
        'Power_lag_48': [float(input("Power 48h ago (kW): "))]
    })[assets['features']]

# Interactive test
test_sample = create_test_sample()
print("Prediction:", predict_failure(test_sample)[0])


# In[ ]:





# In[43]:


# Test edge cases (low power, high vibration, etc.)
edge_cases = pd.DataFrame({
    'LV ActivePower (kW)': [100, 500, 50],  # Extremely low/high values
    'Wind Speed (m/s)': [8.0, 3.0, 10.0],   # Storm/calm conditions
    'Efficiency': [0.2, 0.95, 0.1],         # Abnormal efficiency
    'Power_6h_avg': [90, 480, 40],          # Rolling averages
    'Power_6h_std': [30, 5, 60],            # High variability
    'Power_lag_6': [80, 490, 30],           # Historical lows/highs
    'Power_lag_48': [70, 500, 20]           # Older history
})[assets['features']]

print("Edge Case Predictions:")
print(predict_failure(edge_cases))


# In[44]:


# Save test cases to CSV
edge_cases.to_csv('turbine_test_cases.csv', index=False)
print("Saved test cases to turbine_test_cases.csv")


# In[ ]:





# In[ ]:




