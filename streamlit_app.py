import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import joblib

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_fake_news_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

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

def main():
    st.set_page_config(
        page_title="Malaysian Fake News Detector",
        page_icon="🔍",
        layout="centered"
    )
    
    st.title("Malaysian Fake News Detector 🔍")
    
    # Download NLTK data
    download_nltk_data()
    
    # Load pre-trained model
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    # Create text input
    text_input = st.text_area(
        "Enter news article text:",
        height=200,
        placeholder="Paste your news article here..."
    )
    
    if st.button("Detect Fake News", type="primary"):
        if text_input:
            try:
                # Preprocess input text
                processed_text = preprocess_text(text_input)
                
                # Make prediction using the loaded model
                # Note: The model should include both the classifier and vectorizer
                prediction = model.predict([processed_text])[0]
                probability = model.predict_proba([processed_text])[0]
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
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main() 