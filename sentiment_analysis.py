
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Ensure that NLTK data is downloaded (e.g., stopwords, punkt)
nltk.download('stopwords')
nltk.download('punkt')

# Load your dataset (Sentiment140 or Twitter data)
# Example with Sentiment140 dataset
df = pd.read_csv('data/sentiment140.csv', encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Preprocessing
df['text'] = df['text'].apply(lambda x: ' '.join(word for word in x.split() if word[0] != '@'))  # Remove @mentions
df['text'] = df['text'].apply(lambda x: ' '.join(word for word in x.split() if word[0] != '#'))  # Remove hashtags

# Clean text - Remove special characters, convert to lowercase, etc.
df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)  # Remove URLs
df['text'] = df['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)  # Remove non-alphabetical characters
df['text'] = df['text'].str.lower()

# Define features and labels
X = df['text']
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report: 
{classification_report(y_test, y_pred)}')

# Plot Sentiment Distribution (Positive, Negative, Neutral)
plt.figure(figsize=(8,6))
df['target'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['green', 'red', 'blue'], labels=['Positive', 'Negative', 'Neutral'])
plt.title("Sentiment Distribution")
plt.ylabel('')
plt.show()
