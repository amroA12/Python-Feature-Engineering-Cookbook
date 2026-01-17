import pandas as pd

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='train')
df = pd.DataFrame(data.data, columns=['text'])
df.head()

print(df["text"][1])

df['num_char'] = df['text'].str.len()
df.head()

df['num_char'] = df['text'].str.strip().str.len()
df.head()

df["text"].loc[1].split()

df['num_words'] = df['text'].str.split().str.len()
df.head()

df['num_words'] = df['text'].str.strip().str.split().str.len()
df.head()

df['num_vocab'] = df['text'].str.split().apply(set).str.len()
df.head()

df['num_vocab'] = df['text'].str.lower().str.split().apply(set).str.len()
df.head()

df['lexical_div'] = df['num_words'] / df['num_vocab']
df.head()

df['ave_word_length'] = df['num_char'] / df['num_words']
df.head()

data.target_names                                               

import matplotlib.pyplot as plt

df['target'] = data.target

def plot_features(df, text_var):

    nb_rows = 5
    nb_cols = 4
    
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(12, 12))
    plt.subplots_adjust(wspace=None, hspace=0.4)

    n = 0
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            axs[i, j].hist(df[df.target==n][text_var], bins=30)
            axs[i, j].set_title(text_var + ' | ' + str(n))
            n += 1
    plt.show()
    
plot_features(df, 'num_words')

