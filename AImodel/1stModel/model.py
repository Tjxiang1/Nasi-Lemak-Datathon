import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Load the trained components
model = joblib.load('engagement_classifier.joblib')
scaler = joblib.load('scaler.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict(tag):
    """
    Predict engagement level from a single tag.
    
    Args:
        tag: String tag (e.g., "hair", "beauty", "makeup")
    
    Returns:
        dict: Prediction result with engagement level and probability
    """
    
    try:
        # Step 1: Vectorize the tag using TF-IDF (produces 500 features)
        tag_vectorized = tfidf_vectorizer.transform([tag])
        tag_dense = tag_vectorized.toarray()
        
        # Step 2: Create the 7 numeric features that match your training
        # These are the exact features from your feature engineering:
        # ["likes_per_view", "comments_per_view", "view_growth", "like_growth", "comment_growth", "month_sin", "month_cos"]
        
        # Default values for numeric features (you can adjust these)
        numeric_features = np.array([
            0.05,   # likes_per_view (5% like rate)
            0.01,   # comments_per_view (1% comment rate)
            0.0,    # view_growth (no growth data available)
            0.0,    # like_growth (no growth data available)
            0.0,    # comment_growth (no growth data available)
            0.0,    # month_sin (default month)
            1.0     # month_cos (default month)
        ])
        
        # Combine TF-IDF features with numeric features (exactly like your training)
        combined_features = np.concatenate([tag_dense[0], numeric_features])
        features = combined_features.reshape(1, -1)
        
        # Step 3: Try to scale if scaler is compatible
        try:
            features_scaled = scaler.transform(features)
            final_features = features_scaled
        except ValueError:
            # If scaling fails, use raw features
            print(f"Warning: Scaler incompatible, using raw features")
            final_features = features
        
        # Step 4: Make prediction
        prediction = model.predict(final_features)[0]
        probability = model.predict_proba(final_features)[0]
        
        return {
            'tag': tag,
            'prediction': 'High Engagement' if prediction == 1 else 'Low Engagement',
            'probability_high': round(probability[1], 3),
            'probability_low': round(probability[0], 3),
            'confidence': 'High' if max(probability) > 0.7 else 'Medium' if max(probability) > 0.6 else 'Low'
        }
        
    except Exception as e:
        return {
            'tag': tag,
            'error': str(e),
            'prediction': 'Error',
            'probability_high': 0,
            'probability_low': 0,
            'confidence': 'Error'
        }

def predict_detailed(tag, likes_per_view=0.05, comments_per_view=0.01, 
                    view_growth=0.0, like_growth=0.0, comment_growth=0.0, 
                    month_sin=0.0, month_cos=1.0):
    """
    Predict engagement with detailed numeric features.
    
    Args:
        tag: String tag
        likes_per_view: Like rate (default: 0.05 = 5%)
        comments_per_view: Comment rate (default: 0.01 = 1%)
        view_growth: View growth rate (default: 0.0)
        like_growth: Like growth rate (default: 0.0)
        comment_growth: Comment growth rate (default: 0.0)
        month_sin: Month sine value (default: 0.0)
        month_cos: Month cosine value (default: 1.0)
    
    Returns:
        dict: Prediction result
    """
    
    try:
        # Step 1: Vectorize the tag using TF-IDF
        tag_vectorized = tfidf_vectorizer.transform([tag])
        tag_dense = tag_vectorized.toarray()
        
        # Step 2: Use provided numeric features
        numeric_features = np.array([
            likes_per_view,
            comments_per_view,
            view_growth,
            like_growth,
            comment_growth,
            month_sin,
            month_cos
        ])
        
        # Combine features
        combined_features = np.concatenate([tag_dense[0], numeric_features])
        features = combined_features.reshape(1, -1)
        
        # Step 3: Scale features
        try:
            features_scaled = scaler.transform(features)
            final_features = features_scaled
        except ValueError:
            print(f"Warning: Scaler incompatible, using raw features")
            final_features = features
        
        # Step 4: Make prediction
        prediction = model.predict(final_features)[0]
        probability = model.predict_proba(final_features)[0]
        
        return {
            'tag': tag,
            'prediction': 'High Engagement' if prediction == 1 else 'Low Engagement',
            'probability_high': round(probability[1], 3),
            'probability_low': round(probability[0], 3),
            'confidence': 'High' if max(probability) > 0.7 else 'Medium' if max(probability) > 0.6 else 'Low',
            'features_used': {
                'likes_per_view': likes_per_view,
                'comments_per_view': comments_per_view,
                'view_growth': view_growth,
                'like_growth': like_growth,
                'comment_growth': comment_growth,
                'month_sin': month_sin,
                'month_cos': month_cos
            }
        }
        
    except Exception as e:
        return {
            'tag': tag,
            'error': str(e),
            'prediction': 'Error',
            'probability_high': 0,
            'probability_low': 0,
            'confidence': 'Error'
        }

def predict_multiple(tags):
    """
    Predict engagement for multiple tags.
    
    Args:
        tags: List of tag strings
    
    Returns:
        list: List of prediction results
    """
    results = []
    for tag in tags:
        result = predict(tag)
        results.append(result)
    return results

# Example usage
if __name__ == "__main__":
    # Test with simple tag prediction
    result = predict("hair")
    print("Simple tag prediction:")
    print(f"Result: {result}")
    
    # Test with detailed prediction
    result_detailed = predict_detailed("hair", likes_per_view=0.08, comments_per_view=0.02)
    print(f"\nDetailed prediction:")
    print(f"Result: {result_detailed}")
    
    # Test with multiple tags
    tags = ["beauty", "makeup", "skincare", "fashion"]
    results = predict_multiple(tags)
    print(f"\nMultiple tag predictions:")
    for result in results:
        print(f"- {result['tag']}: {result['prediction']} (confidence: {result['confidence']})")
