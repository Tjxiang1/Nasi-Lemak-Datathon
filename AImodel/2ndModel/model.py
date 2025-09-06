import joblib

model = joblib.load('XGBoost_trend_classifier.pkl')
encoder = joblib.load('trend_label_encoder.pkl')

input = [1]
encoded_data = encoder.transform(input)

pred_encoded = model.predict(encoded_data)
pred = encoder.inverse_transform(pred_encoded)

print(pred)
