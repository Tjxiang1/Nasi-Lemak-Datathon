import joblib
import pandas as pd 
import numpy as np 

model = joblib.load('XGBoost_trend_classifier.pkl')
label_encoder = joblib.load('trend_label_encoder.pkl')
df = pd.read_csv('model2.csv')


def predict_trend(hashtag: str, year: int, week: int, raw_data: pd.DataFrame) -> dict:
    """
    Predicts the trend phase for a given hashtag and time window.
    
    Parameters:
        hashtag (str): The hashtag to analyze (e.g. "#beauty")
        year (int): The year of interest (e.g. 2025)
        week (int): The week number (e.g. 36)
        raw_data (pd.DataFrame): Full dataset with timestamped hashtag metrics
    
    Returns:
        dict: Contains predicted label and confidence scores
    """
    
    # Step 1: Filter and preprocess data for the given hashtag and time
    df_filtered = raw_data[
        (raw_data["allTags_cleaned"].str.contains(hashtag, case=False)) &
        ((raw_data["year"] < year) | ((raw_data["year"] == year) & (raw_data["week"] <= week)))
    ].copy()
    
    if df_filtered.empty:
        return {"error": "No data available for this hashtag and time window."}
    
    # Step 2: Feature engineering (must match training pipeline)
    # Example: rolling metrics, IQR labeling, encoding, scaling, etc.
    # This is a placeholder — replace with your actual pipeline
    X_new = engineer_features(df_filtered, year, week)  # <- your custom function
    
    # Step 3: Predict and decode label
    pred_numeric = model.predict(X_new)[0]
    pred_label = label_encoder.inverse_transform([pred_numeric])[0]
    
    # Step 4: Confidence scores
    probs = model.predict_proba(X_new)[0]
    class_probs = dict(zip(label_encoder.classes_, probs))
    
    return {
        "hashtag": hashtag,
        "year": year,
        "week": week,
        "predicted_phase": pred_label,
        "confidence": {k: f"{v:.2%}" for k, v in class_probs.items()}
    }


def engineer_features(df: pd.DataFrame, year: int, week: int) -> pd.DataFrame:
    """
    Transforms raw hashtag-week data into model-ready features.
    
    Parameters:
        df (pd.DataFrame): Filtered data for a single hashtag up to the target week
        year (int): Target year
        week (int): Target week number
    
    Returns:
        pd.DataFrame: One-row DataFrame with engineered features
    """
    
    # Step 1: Ensure datetime format
    df["date"] = pd.to_datetime((df["year"]).astype(int).astype(str) + "-" + df["week"].astype(str) + "-1", format="%Y-%W-%w")
    df = df.sort_values("date")
    
    # Step 2: Aggregate weekly metrics
    weekly = df.groupby("date").agg({
        "post_count": "sum",
        "engagement": "sum",
        "unique_users": "nunique"
    }).reset_index()
    
    # Step 3: Rolling metrics (3-week window)
    weekly["rolling_engagement_mean"] = weekly["engagement"].rolling(window=3).mean()
    weekly["rolling_engagement_std"] = weekly["engagement"].rolling(window=3).std()
    weekly["momentum"] = weekly["engagement"].pct_change(periods=1)
    
    # Step 4: Hashtag age
    weekly["hashtag_age"] = (weekly["date"] - weekly["date"].min()).dt.days // 7
    
    # Step 5: Extract target week row
    target_date = pd.to_datetime(f"{int(year)}-{week}-1", format="%Y-%W-%w")
    row = weekly[weekly["date"] == target_date]
    
    if row.empty:
        raise ValueError("Target week not found in data.")
    
    # Step 6: Final feature selection
    features = row[[
        "rolling_engagement_mean",
        "rolling_engagement_std",
        "momentum",
        "hashtag_age"
    ]].copy()
    
    # Step 7: Add seasonal indicator
    features["week_of_year"] = week
    
    # Step 8: Fill missing values (e.g. early weeks)
    features.fillna(0, inplace=True)
    
    return features


print(predict_trend("#hair", 2020, 17, df))





