# Malaysian Fake News Detection

This project implements a machine learning-based fake news detection system specifically for news articles in Bahasa Malaysia. The system uses natural language processing and machine learning techniques to classify news articles as either genuine or fake.

## Features

- Machine learning model trained on Malaysian news dataset
- Real-time fake news detection
- Support for Bahasa Malaysia text
- Text analysis and key terms extraction
- Confidence scores for predictions
- User-friendly Streamlit interface

## Live Demo

Visit the live application at: https://malaysian-fake-news-detection.streamlit.app

## Tech Stack

- **Frontend & Backend**: Streamlit
- **ML & Data Processing**: 
  - scikit-learn
  - NLTK
  - pandas
  - numpy

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/jorre-bot/malaysian-fake-news-detection.git
cd malaysian-fake-news-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The application will be available at http://localhost:8501

## Deployment

This application is deployed using Streamlit Cloud. To deploy your own version:

1. Fork this repository
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your forked repository
4. Select `streamlit_app.py` as the main file
5. Deploy!

## Model Information

The fake news detection model uses:
- Text preprocessing with NLTK
- TF-IDF vectorization
- Naive Bayes classification
- English stopwords removal (can be extended for Bahasa Malaysia)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License 