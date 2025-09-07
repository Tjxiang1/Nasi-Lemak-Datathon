import joblib
import numpy as np 
from scipy.sparse import hstack

def main(tag):        
    bundle = joblib.load('firstModel/trend_model_package.joblib')
    model = bundle['model']
    vectorizer = bundle['vectorizer']
    scaler = bundle['scaler']
    tag_trends = bundle['tag_trends']

    tag_features = vectorizer.transform([tag.lower()])
    tag_row = tag_trends[tag_trends['tag'] == tag]

    if not tag_row.empty:
        default_numeric = tag_row[['view_growth_rate', 'engagement_growth_rate', 'total_views', 
                        'total_engagement', 'months_active', 'avg_duration']].fillna(0)
        # ...existing code...
    tag_features = vectorizer.transform([tag.lower()])
    tag_row = tag_trends[tag_trends['tag'] == tag.lower()]  # Ensure lowercase match

    if not tag_row.empty:
        default_numeric = tag_row[['view_growth_rate', 'engagement_growth_rate', 'total_views', 
                        'total_engagement', 'months_active', 'avg_duration']].fillna(0)

    else:
        print(f"No historical data for tag: {tag}")
        return

    numeric_scaled = scaler.transform(default_numeric)  # Now feature names are preserved

    # Combine features
    X = hstack([tag_features, numeric_scaled])

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
        #'confidence': max(probability),
        'growing_probability': float(round((probability[1]), 2)),
        'decaying_probability': float(round((probability[0]), 2))
    }

    return(result)

if __name__ == '__main__':
    arr = ['hair', 'makeup', 'shorts', 'skincare', 'beauty', 'viral']
    for e in arr:
        main(e)