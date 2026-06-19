---
name: train-model
description: Engineer features and train a delay-prediction model. Use when the user wants to build or evaluate an ML model on the flight data.
disable-model-invocation: true
---

Train a flight delay prediction model. Work inside `notebooks/01_flight_analysis.ipynb` (feature-engineering and modelling sections) unless the user specifies a different file. Accept $ARGUMENTS to override target (regression/classification) or model type.

Steps:
1. **Load & filter**: load with FLIGHT_DTYPES, drop CANCELLED==1 rows.
2. **Feature engineering**:
   - Cyclical encoding for MONTH, DAY_OF_WEEK, hour (sin/cos)
   - SCHEDULED_DEPARTURE_HOUR = SCHEDULED_DEPARTURE // 100
   - ROUTE = ORIGIN_AIRPORT + '_' + DESTINATION_AIRPORT
   - Label-encode AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT
   - Drop leakage columns: AIR_SYSTEM_DELAY, SECURITY_DELAY, AIRLINE_DELAY, LATE_AIRCRAFT_DELAY, WEATHER_DELAY, DEPARTURE_TIME, WHEELS_OFF, WHEELS_ON, TAXI_IN, TAXI_OUT, ELAPSED_TIME, AIR_TIME, ARRIVAL_TIME
3. **Split**: 80/20 train/test, stratify on MONTH.
4. **Baseline**: DummyRegressor (mean) or DummyClassifier (most-frequent).
5. **Model**: default to RandomForestRegressor (n_estimators=100, n_jobs=-1) for regression or RandomForestClassifier for classification (>15 min = late). If user specifies another model (XGBoost, LightGBM, LinearRegression), use that.
6. **Evaluate**:
   - Regression: RMSE, MAE, R²
   - Classification: accuracy, precision, recall, F1, confusion matrix
7. **Feature importance**: horizontal bar chart of top 15 features.
8. Print a 3-bullet summary: best metric vs baseline, top 3 predictive features, suggested next step.
