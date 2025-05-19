# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# import torch
# import pandas as pd
# import numpy as np
# import os

# app = Flask(__name__)
# CORS(app)

# # Configuration
# TRUSTED_SOURCES = ['washington post', 'reuters', 'associated press', 'white house', 'bloomberg']
# POLITICAL_KEYWORDS = ['biden', 'trump', 'congress', 'senate', 'house']
# MAX_LENGTH = 256

# # Load model and tokenizer
# model_path = 'model/distilbert-fakenews'
# tokenizer = DistilBertTokenizer.from_pretrained(model_path)
# model = DistilBertForSequenceClassification.from_pretrained(model_path)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# model.eval()

# # Optional: Validate with real news dataset
# def validate_with_kaggle_data(text):
#     """Check if text matches known real news from Kaggle dataset"""
#     try:
#         kaggle_data = pd.read_csv('data/real_news_samples.csv')
#         return text in kaggle_data['text'].values
#     except Exception as e:
#         print(f"Kaggle data validation error: {e}")
#         return False

# # Check for presence of trusted source
# def contains_trusted_source(text):
#     return any(source in text.lower() for source in TRUSTED_SOURCES)

# # Check for political terms
# def is_political(text):
#     return any(keyword in text.lower() for keyword in POLITICAL_KEYWORDS)

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         text = data.get('text', '').strip()

#         if not text:
#             return jsonify({'error': 'Empty text input'}), 400

#         # Rule-based override: Trusted source
#         if contains_trusted_source(text):
#             return jsonify({
#                 'prediction': 'Real',
#                 'confidence': 85,
#                 'note': 'Trusted source identified',
#                 'text': text
#             })

#         # Rule-based override: Verified in dataset
#         if validate_with_kaggle_data(text):
#             return jsonify({
#                 'prediction': 'Real',
#                 'confidence': 90,
#                 'note': 'Matched verified news in dataset',
#                 'text': text
#             })

#         # Preprocess input
#         inputs = tokenizer(
#             text,
#             return_tensors='pt',
#             padding='max_length',
#             truncation=True,
#             max_length=MAX_LENGTH
#         ).to(device)

#         with torch.no_grad():
#             outputs = model(**inputs)

#         # Use softmax for confidence
#         logits = outputs.logits.cpu()
#         probs = torch.nn.functional.softmax(logits, dim=1).numpy()
#         confidence = float(np.max(probs)) * 100
#         prediction = int(np.argmax(probs))

#         # Political content heuristic adjustment
#         if is_political(text) and prediction == 1:
#             confidence = max(50, confidence * 0.7)

#         result = {
#             'prediction': 'Fake' if prediction == 1 else 'Real',
#             'confidence': int(confidence),
#             'text': text,
#             'warnings': []
#         }

#         if confidence < 70:
#             result['warnings'].append('Low confidence - consider longer text with sources')
#         if is_political(text):
#             result['warnings'].append('Political content - predictions may be less reliable')

#         return jsonify(result)

#     except Exception as e:
#         print(f"Server error: {e}")
#         return jsonify({
#             'error': str(e),
#             'note': 'See server logs for details'
#         }), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Configuration for DistilBERT model
DISTILBERT_MODEL_PATH = 'model/distilbert-fakenews'
DISTILBERT_MAX_LENGTH = 256
tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_PATH)
model_distilbert = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_MODEL_PATH)
model_distilbert.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model_distilbert.eval()

# Configuration for TF-IDF model
TFIDF_MODEL_PATH = os.path.join('model', 'fake_news_model.pkl')
TFIDF_VECTORIZER_PATH = os.path.join('model', 'vectorizer.pkl')
model_tfidf = pickle.load(open(TFIDF_MODEL_PATH, 'rb'))
vectorizer = pickle.load(open(TFIDF_VECTORIZER_PATH, 'rb'))

@app.route('/predict/distilbert', methods=['POST'])
def predict_distilbert():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'Empty text input'}), 400

        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=DISTILBERT_MAX_LENGTH
        ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        with torch.no_grad():
            outputs = model_distilbert(**inputs)

        logits = outputs.logits.cpu()
        probs = torch.nn.functional.softmax(logits, dim=1).numpy()
        confidence = float(np.max(probs)) * 100
        prediction = int(np.argmax(probs))

        result = {
            'model': 'DistilBERT',
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': int(confidence),
            'text': text
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/tfidf', methods=['POST'])
def predict_tfidf():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'Empty text input'}), 400

        text_vectorized = vectorizer.transform([text])
        prediction = model_tfidf.predict(text_vectorized)
        confidence = model_tfidf.decision_function(text_vectorized)
        confidence_percent = int(100 * (1 / (1 + abs(confidence[0]))))

        result = {
            'model': 'TF-IDF',
            'prediction': 'Fake' if prediction[0] == 1 else 'Real',
            'confidence': confidence_percent,
            'text': text
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
