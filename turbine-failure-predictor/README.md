# Wind Turbine Failure Prediction System

![Wind Turbine]

## 📦 Production Deployment Checklist

### Prerequisites
- Python 3.8+
- Dependencies: `pandas`, `scikit-learn`, `joblib`, `numpy`

### File Structure
turbine-failure-prediction/
|── turbine_failure_predictor.pkl # Trained model binary
|── feature_list.txt # Required input feature
│── turbine_test_cases.csv # Sample test data
│── T1.csv # Template for new inputs
│── train_model.ipynb # Training notebook
│── deploy.py # Production prediction script
└── README.md # This file


## 🚀 Quick Start
1. Clone this repository
```bash
git clone https://github.com/yourusername/wind-turbine-failure.git
cd wind-turbine-failure

pip install -r requirements.txt

python src/deploy.py data/new_scada_data.csv

[
    'LV ActivePower (kW)',
    'Wind Speed (m/s)',
    'Efficiency',
    'Power_6h_avg',
    'Power_6h_std',
    'Power_lag_6',
    'Power_lag_48'
]


### How to Use This:
1. **Save the file**:
   - In Jupyter: `!echo '<paste entire content above>' > README.md`
   - Or create new file in your IDE/text editor

2. **Customize**:
   - Replace placeholder links/emails
   - Update file paths if your structure differs
   - Add your actual model performance metrics

3. **Keep updated**:
   - Version history at bottom
   - Add new issues/solutions as encountered

