import joblib
import numpy as np 
from scipy.sparse import hstack

def main(tag):
        
    model = joblib.load('trend_model.joblib')
    scaler = joblib.load('trend_scaler.joblib')
    vectorizer = joblib.load('trend_vectorizer.joblib')
    tag_trends = joblib.load('tag_trends_data.joblib')


    tag_features = vectorizer.transform([tag.lower()])
    tag_row = tag_trends[tag_trends['tag'] == tag]

    if not tag_row.empty:
        default_numeric = tag_row[['view_growth_rate', 'engagement_growth_rate', 'total_views', 
                        'total_engagement', 'months_active', 'avg_duration']].fillna(0).values
    else:
        print(f"No historical data for tag: {tag}")

    # Prepare default numeric features (adjust as needed)
    #default_numeric = np.array([[1, 0, 1000, 0.05, 6, 30]])  #'view_growth_rate', 'engagement_growth_rate', 'total_views', 'total_engagement', 'months_active', 'avg_duration'

    numeric_scaled = scaler.transform(default_numeric)

    # Combine features
    X = hstack([tag_features, numeric_scaled])

    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    result = {
        'tag': tag,
        'trend': 'Growing' if prediction == 1 else 'Decaying',
        'confidence': max(probability),
        'growing_probability': probability[1],
        'decaying_probability': probability[0]
    }

    print(result)

if __name__ == '__main__':
    main('shorts')