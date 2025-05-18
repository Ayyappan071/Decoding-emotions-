import tweepy
import pandas as pd
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Twitter API credentials
consumer_key = 'YOUR_CONSUMER_KEY'
consumer_secret = 'YOUR_CONSUMER_SECRET'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

# Authenticate
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Function to clean tweet text
def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#[A-Za-z0-9_]+|[^A-Za-z\s]", "", text)
    text = " ".join([word for word in text.lower().split() if word not in stop_words])
    return text

# Fetch tweets
query = "#AI -filter:retweets"
tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode='extended').items(100)
data = [{"text": tweet.full_text} for tweet in tweets]
df = pd.DataFrame(data)

# Clean text
df['clean_text'] = df['text'].apply(clean_text)

# Load pre-trained BERT model for emotion detection
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Emotion labels
emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# Analyze emotions
def get_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = softmax(logits.numpy()[0])
    return emotion_labels[scores.argmax()]

df['emotion'] = df['clean_text'].apply(get_emotion)

# Visualization
emotion_counts = df['emotion'].value_counts()
emotion_counts.plot(kind='bar', title='Emotion Distribution in Tweets', color='skyblue')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Save to CSV
df.to_csv("twitter_emotion_analysis.csv", index=False)
