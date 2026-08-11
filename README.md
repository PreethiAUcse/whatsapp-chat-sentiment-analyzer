# WhatsApp Chat and Sentiment Analyzer

A Streamlit-based web application that analyzes exported WhatsApp chat data and provides meaningful insights through statistics, visualizations, and sentiment analysis.

## Features

- Total messages, words, media and links
- Most active users
- Word cloud of frequently used words
- Emoji usage analysis
- Monthly message timeline
- Daily message timeline
- Activity heatmap
- Positive, negative and neutral sentiment analysis

## Technologies Used

- Python
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- WordCloud
- URLExtract
- Emoji
- NLTK
- VADER Sentiment Analysis

## System Workflow

```text
WhatsApp Chat (.txt)
        ↓
Data Preprocessing
        ↓
Structured DataFrame
        ↓
Data Analysis
        ↓
Visualizations
        ↓
Sentiment Analysis
        ↓
Streamlit Dashboard

## Project Structure:
whatsapp-chat-sentiment-analyzer/
│
├── app.py
├── helper.py
├── preprocessor.py
├── requirements.txt
├── README.md
└── sample_data/
    └── sample_chat.txt

## How to Run
Install the required dependencies: pip install -r requirements.txt
Run the Streamlit application: python -m streamlit run app.py
(The application opens in a web browser.)

## How It Works
-The user uploads an exported WhatsApp .txt chat file.
-The raw chat is processed and converted into structured data.
-Message statistics and user activity are calculated.
-Visualizations such as timelines, word clouds and heatmaps are generated.
-Sentiment analysis classifies messages as positive, negative or neutral.
-The results are displayed through the Streamlit dashboard.
-Sentiment Analysis

## The application analyzes the sentiment of messages and presents their distribution as:
-Positive
-Negative
-Neutral

## Limitations:
-Accuracy depends on the WhatsApp chat export format.
-Sarcasm and context may not always be interpreted correctly.
-Sentiment analysis may be less accurate for regional and mixed languages.

## Future Scope:
-Multilingual sentiment analysis
-Advanced NLP models
-Real-time analysis
-Support for other messaging platforms
-Improved emotion detection

