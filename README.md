# Malaysian Fake News Detection

This project implements a machine learning-based fake news detection system specifically for news articles in Bahasa Malaysia. The system uses natural language processing and deep learning techniques to classify news articles as either genuine or fake.

## Features

- Machine learning model trained on Malaysian news dataset
- Web interface for real-time fake news detection
- Support for Bahasa Malaysia text
- REST API endpoints for integration
- Modern React-based frontend
- Flask backend server

## Tech Stack

- **Frontend**: React.js, TailwindCSS
- **Backend**: Python Flask
- **ML**: scikit-learn, TensorFlow, Malaya NLP
- **Database**: SQLite (for caching results)

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/jorre-bot/malaysian-fake-news-detection.git
cd malaysian-fake-news-detection
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
```

4. Start the backend server:
```bash
python app.py
```

5. Start the frontend development server:
```bash
cd frontend
npm start
```

The application will be available at http://localhost:3000

## API Documentation

### POST /api/detect
Endpoint for fake news detection

Request body:
```json
{
    "text": "Your news article text here"
}
```

Response:
```json
{
    "prediction": "FAKE/REAL",
    "confidence": 0.95,
    "analysis": {
        "key_features": [],
        "sentiment": ""
    }
}
```

## Model Information

The fake news detection model is built using a combination of:
- Text preprocessing using Malaya NLP library
- TF-IDF vectorization
- Deep learning classification using BERT

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 