import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def parse_tag_list(value):
    """Parse tag lists from string format"""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(tag).strip().lower() for tag in parsed if str(tag).strip()]
    except Exception:
        pass
    return [s.strip().lower() for s in str(value).split(',') if s.strip()]

def calculate_tag_trends(df):
    """Calculate growth/decay trends for each tag"""
    print("📊 Calculating tag trends...")
    
    # Parse tags and explode the dataframe
    df['tag_list'] = df['allTags_cleaned'].apply(parse_tag_list)
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    
    # Create month-year for trend analysis
    df['month_year'] = df['publishedAt'].dt.to_period('M')
    
    # Explode tags to get one row per tag per video
    exploded = df.explode('tag_list')
    exploded = exploded[exploded['tag_list'].notna() & (exploded['tag_list'] != '')]
    
    # Calculate monthly metrics for each tag
    monthly_metrics = exploded.groupby(['tag_list', 'month_year']).agg({
        'viewCount': 'sum',
        'likeCount': 'sum', 
        'commentCount': 'sum',
        'contentDuration(seconds)': 'mean',
        'likes_per_view': 'mean',
        'comments_per_view': 'mean'
    }).reset_index()
    
    # Calculate trend for each tag
    tag_trends = []
    
    for tag in monthly_metrics['tag_list'].unique():
        tag_data = monthly_metrics[monthly_metrics['tag_list'] == tag].sort_values('month_year')
        
        if len(tag_data) < 3:  # Need at least 3 months for trend analysis
            continue
            
        # Calculate growth rates
        recent_months = tag_data.tail(3)  # Last 3 months
        older_months = tag_data.head(3)   # First 3 months
        
        recent_avg_views = recent_months['viewCount'].mean()
        older_avg_views = older_months['viewCount'].mean()
        
        recent_avg_engagement = recent_months['likes_per_view'].mean()
        older_avg_engagement = older_months['likes_per_view'].mean()
        
        # Calculate growth rates
        view_growth_rate = (recent_avg_views - older_avg_views) / (older_avg_views + 1)
        engagement_growth_rate = (recent_avg_engagement - older_avg_engagement) / (older_avg_engagement + 0.001)
        
        # Determine trend (1 = growing, 0 = decaying)
        # A tag is considered growing if views and engagement are increasing
        is_growing = 1 if (view_growth_rate > 0.1 and engagement_growth_rate > 0.05) else 0
        
        # Calculate additional features
        total_views = tag_data['viewCount'].sum()
        total_engagement = tag_data['likes_per_view'].mean()
        months_active = len(tag_data)
        avg_duration = tag_data['contentDuration(seconds)'].mean()
        
        tag_trends.append({
            'tag': tag,
            'is_growing': is_growing,
            'view_growth_rate': view_growth_rate,
            'engagement_growth_rate': engagement_growth_rate,
            'total_views': total_views,
            'total_engagement': total_engagement,
            'months_active': months_active,
            'avg_duration': avg_duration,
            'recent_views': recent_avg_views,
            'older_views': older_avg_views
        })
    
    return pd.DataFrame(tag_trends)

def prepare_features(tag_trends_df):
    """Prepare features for the trend prediction model"""
    print("🔧 Preparing features...")
    
    # Create TF-IDF features for tags
    vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    tag_features = vectorizer.fit_transform(tag_trends_df['tag'])
    
    # Create numeric features
    numeric_features = tag_trends_df[[
        'view_growth_rate', 'engagement_growth_rate', 'total_views', 
        'total_engagement', 'months_active', 'avg_duration'
    ]].fillna(0)
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_features)
    
    # Combine features
    from scipy.sparse import hstack
    X = hstack([tag_features, numeric_scaled])
    
    y = tag_trends_df['is_growing']
    
    return X, y, vectorizer, scaler

def train_trend_model():
    """Train the trend prediction model"""
    print("🚀 Starting trend prediction model training...")
    
    # Load data
    print("📁 Loading data...")
    df = pd.read_csv('model2.csv')
    print(f"Loaded {len(df)} records")
    
    # Calculate trends
    tag_trends = calculate_tag_trends(df)
    print(f"Calculated trends for {len(tag_trends)} tags")
    
    # Check class distribution
    print(f"Growing tags: {tag_trends['is_growing'].sum()}")
    print(f"Decaying tags: {(tag_trends['is_growing'] == 0).sum()}")
    
    # Prepare features
    X, y, vectorizer, scaler = prepare_features(tag_trends)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Accuracy: {accuracy:.3f}")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Decaying', 'Growing']))
    
    # Save model and components
    joblib.dump(model, 'trend_model.joblib')
    joblib.dump(vectorizer, 'trend_vectorizer.joblib')
    joblib.dump(scaler, 'trend_scaler.joblib')
    joblib.dump(tag_trends, 'tag_trends_data.joblib')
    
    print("💾 Model and components saved!")
    
    return model, vectorizer, scaler, tag_trends

def predict_tag_trend(tag, model, vectorizer, scaler):
    """Predict if a tag is growing or decaying"""
    
    # Create features for the input tag
    tag_features = vectorizer.transform([tag.lower()])
    
    # For numeric features, we'll use default values since we don't have historical data
    # In a real scenario, you'd calculate these from recent data
    default_numeric = np.array([[0, 0, 1000, 0.05, 6, 30]])  # Default values
    numeric_scaled = scaler.transform(default_numeric)
    
    # Combine features
    from scipy.sparse import hstack
    X = hstack([tag_features, numeric_scaled])
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    return {
        'tag': tag,
        'trend': 'Growing' if prediction == 1 else 'Decaying',
        'confidence': max(probability),
        'growing_probability': probability[1],
        'decaying_probability': probability[0]
    }

if __name__ == "__main__":
    # Train the model
    model, vectorizer, scaler, tag_trends = train_trend_model()
    
    # Test predictions
    print("\n🧪 Testing predictions:")
    test_tags = ['makeup', 'skincare', 'hair', 'fashion', 'beauty']
    
    for tag in test_tags:
        result = predict_tag_trend(tag, model, vectorizer, scaler)
        print(f"Tag: {tag} → {result['trend']} (confidence: {result['confidence']:.3f})")
