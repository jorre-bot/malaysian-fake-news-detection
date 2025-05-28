from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import malaya
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load and preprocess the dataset
def load_data():
    df = pd.read_csv('combined_news_articles.csv')
    return df

# Train the model
def train_model():
    df = load_data()
    
    # Initialize Malaya's pretrained model for text preprocessing
    tokenizer = malaya.preprocessing.Tokenizer()
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(lambda x: ' '.join(tokenizer.tokenize(str(x))))
    
    # Create TF-IDF features
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['processed_text'])
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Save the model and vectorizer
    joblib.dump(model, 'model.pkl')
    joblib.dump(tfidf, 'tfidf.pkl')
    
    return model, tfidf

# Load or train the model
if not os.path.exists('model.pkl'):
    model, tfidf = train_model()
else:
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf.pkl')

@app.route('/api/detect', methods=['POST'])
def detect_fake_news():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess the input text
        tokenizer = malaya.preprocessing.Tokenizer()
        processed_text = ' '.join(tokenizer.tokenize(text))
        
        # Transform text using saved vectorizer
        text_features = tfidf.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_features)[0]
        probability = model.predict_proba(text_features)[0]
        
        # Get confidence score
        confidence = float(max(probability))
        
        # Get sentiment analysis
        sentiment_analyzer = malaya.sentiment.transformer(model='bert')
        sentiment = sentiment_analyzer(text)[0]
        
        return jsonify({
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': confidence,
            'analysis': {
                'sentiment': sentiment['label'],
                'key_features': []  # Add key features that influenced the decision
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 