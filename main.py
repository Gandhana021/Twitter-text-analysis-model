#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
import nltk

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/twitter.csv")
print(data.head())


# In[4]:


pip install emoji


# In[4]:


import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  # Import word_tokenize for advanced tokenization
import emoji  # Import the emoji library for emoji handling


nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    # Tokenize the text using word_tokenize
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopword]

    # Convert emojis to text
    text = emoji.demojize(' '.join(tokens))

    # Extract mentions and hashtags
    mentions = re.findall(r'@(\w+)', text)
    hashtags = re.findall(r'#(\w+)', text)

    # Stem the words
    tokens = [stemmer.stem(word) for word in tokens]

    text = " ".join(tokens)
    return text

data["tweet"] = data["tweet"].apply(clean)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["tweet"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["tweet"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["tweet"]]

data = data[["tweet", "Positive", "Negative", "Neutral"]]
print(data.head())

# Continue with the rest of your code, including sentiment analysis and visualization.


# In[6]:


pip install scikit-learn


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer


# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data["tweet"])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
data = pd.concat([data, tfidf_df], axis=1)



# In[6]:


data = data[["tweet", "Positive", 
             "Negative", "Neutral"]]
print(data.head())


# In[7]:


x = sum(data["Positive"])
y = sum(data["Negative"])
z = sum(data["Neutral"])

def sentiment_score(a, b, c):
    if (a>b) and (a>c):
        print("Positive 😊 ")
    elif (b>a) and (b>c):
        print("Negative 😠 ")
    else:
        print("Neutral 🙂 ")
sentiment_score(x, y, z)


# In[8]:


print("Positive: ", x)
print("Negative: ", y)
print("Neutral: ", z)


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


# Assuming you have "Positive," "Negative," and "Neutral" columns in your DataFrame
sentiment_counts = pd.concat([data["Positive"], data["Negative"], data["Neutral"]]).value_counts()

# Select the top N sentiment scores
top_N = 10
sentiment_counts = sentiment_counts.head(top_N)

plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title(f"Top {top_N} Sentiment Scores Distribution")
plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.show()


# In[ ]:




