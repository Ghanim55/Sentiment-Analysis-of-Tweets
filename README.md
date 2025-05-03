
# Sentiment Analysis of Tweets

## Project Overview:
This project implements a Sentiment Analysis model to classify tweets as **Positive**, **Negative**, or **Neutral** based on their content. The model uses the **Sentiment140** dataset or Twitter data for training.

## Setup Instructions:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Sentiment-Analysis-of-Tweets.git
   cd Sentiment-Analysis-of-Tweets
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset (Sentiment140 or Twitter data) and place it in the `data/` folder.

4. Run the sentiment analysis script:
   ```bash
   python sentiment_analysis.py
   ```

## Features:
- Data Preprocessing (Removing stopwords, mentions, special characters)
- TF-IDF for feature extraction
- Naive Bayes for sentiment classification
- Sentiment distribution visualization (Positive, Negative, Neutral)

## Results:
After training, the model will print the accuracy and classification report, and display a pie chart of sentiment distribution.

## License:
This project is licensed under the MIT License.
