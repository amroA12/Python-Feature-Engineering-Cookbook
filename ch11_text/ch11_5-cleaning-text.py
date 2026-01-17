import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='train')
df = pd.DataFrame(data.data, columns=['text'])
df.head()

print(df['text'][10])

df["text"] = df['text'].str.replace(r'[^\w\s]','', regex=True)

print(df['text'][10])

import string

df['text'] = df['text'].str.replace('[{}]'.format(string.punctuation), '', regex=True)

string.punctuation

df['text'] = df['text'].str.replace(r'\d+', '', regex=True)

print(df['text'][10])

df['text'] = df['text'].str.lower()

print(df['text'][10])

stopwords.words('english')

def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    text = [word for word in text.split() if word not in stop]
    text = ' '.join(x for x in text)
    return text

remove_stopwords(df['text'][10])

df['text'] = df['text'].apply(remove_stopwords)

print(df['text'][10])

stemmer = SnowballStemmer("english")

stemmer.stem('running')

def stemm_words(text):
    text = [stemmer.stem(word) for word in text.split()]
    text = ' '.join(x for x in text)
    return text

stemm_words(df['text'][10])

df['text'] = df['text'].apply(stemm_words)

print(df['text'][10])

