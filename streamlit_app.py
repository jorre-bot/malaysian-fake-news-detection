import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('combined_news_articles.csv')
    return df

# Preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

# Train the model
@st.cache_resource
def train_model(df):
    # Preprocess the text data
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Create TF-IDF features
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['processed_text'])
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    return model, tfidf

def main():
    st.set_page_config(
        page_title="Malaysian Fake News Detector",
        page_icon="🔍",
        layout="centered"
    )
    
    st.title("Malaysian Fake News Detector 🔍")
    
    # Download NLTK data
    download_nltk_data()
    
    # Load data and train model
    df = load_data()
    model, tfidf = train_model(df)
    
    # Create text input
    text_input = st.text_area(
        "Enter news article text:",
        height=200,
        placeholder="Paste your news article here..."
    )
    
    if st.button("Detect Fake News", type="primary"):
        if text_input:
            # Preprocess input text
            processed_text = preprocess_text(text_input)
            
            # Transform text using saved vectorizer
            text_features = tfidf.transform([processed_text])
            
            # Make prediction
            prediction = model.predict(text_features)[0]
            probability = model.predict_proba(text_features)[0]
            confidence = float(max(probability))
            
            # Display results
            st.markdown("### Analysis Result")
            
            # Create columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Prediction:**")
                st.markdown("**Confidence:**")
            
            with col2:
                if prediction == 1:
                    st.markdown("🚫 **FAKE**")
                else:
                    st.markdown("✅ **REAL**")
                st.markdown(f"**{confidence*100:.2f}%**")
            
            # Additional analysis
            st.markdown("### Text Analysis")
            
            # Word count
            word_count = len(text_input.split())
            st.info(f"Word count: {word_count}")
            
            # Display most important words
            if word_count > 0:
                st.markdown("#### Key Terms")
                words = processed_text.split()[:10]
                st.write(", ".join(words))
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main() 