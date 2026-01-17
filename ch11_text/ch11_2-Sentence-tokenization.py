import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from sklearn.datasets import fetch_20newsgroups

text = """
The alarm rang at 7 in the morning as it usually did on Tuesdays. She rolled over,
stretched her arm, and stumbled to the button till she finally managed to switch it off.
Reluctantly, she got up and went for a shower. The water was cold as the day before the engineers
did not manage to get the boiler working. Good thing it was still summer.
Upstairs, her cat waited eagerly for his morning snack. Miaow! He voiced with excitement
as he saw her climb the stairs.
"""

sent_tokenize(text)

len(sent_tokenize(text))

data = fetch_20newsgroups(subset='train')
df = pd.DataFrame(data.data, columns=['text'])
df.head()

df = df.loc[1:10]

df['text'] = df['text'].str.split('Lines:', n=1).str[-1]

print(df['text'][1])

df['num_sent'] = df['text'].apply(sent_tokenize).apply(len)

