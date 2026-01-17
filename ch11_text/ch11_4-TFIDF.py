import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

data = fetch_20newsgroups(subset='train')
df = pd.DataFrame(data.data, columns=['text'])
df.head()

df['text'] = (
    df['text']
    .str.replace(r'[^\w\s]', '', regex=True)
    .str.replace(r'\d+', '', regex=True)
)
vectorizer = TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1, 1),min_df=0.05)

vectorizer.fit(df['text'])

X = vectorizer.transform(df['text'])

tfidf = pd.DataFrame(X.toarray(),columns = vectorizer.get_feature_names_out())

tfidf.head()

vectorizer = TfidfVectorizer(lowercase=True,stop_words='english',ngram_range=(1, 2),min_df=0.1)

vectorizer.fit(df['text'])

X = vectorizer.transform(df['text'])

tfidf = pd.DataFrame(X.toarray(),columns = vectorizer.get_feature_names_out())

tfidf.head()

vectorizer.get_feature_names_out()

