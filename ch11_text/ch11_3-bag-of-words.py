import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

data = fetch_20newsgroups(subset='train')
df = pd.DataFrame(data.data, columns=['text'])
df.head()

df['text'] = (
    df['text']
    .str.replace(r'[^\w\s]', '', regex=True)
    .str.replace(r'\d+', '', regex=True)
)

vectorizer = CountVectorizer(lowercase=True,stop_words='english',ngram_range=(1, 1),min_df=0.05)

vectorizer.fit(df['text'])

X = vectorizer.transform(df['text'])

bagofwords = pd.DataFrame(X.toarray(),columns = vectorizer.get_feature_names_out())
bagofwords.head()

vectorizer = CountVectorizer(lowercase=True,stop_words='english',ngram_range=(1, 2),min_df=0.1)

vectorizer.fit(df['text'])

X = vectorizer.transform(df['text'])

bagofwords = pd.DataFrame(X.toarray(),columns = vectorizer.get_feature_names_out())
bagofwords.head()

vectorizer.get_feature_names_out()

