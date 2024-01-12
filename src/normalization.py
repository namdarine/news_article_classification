import pandas as pd
from function import sentence_vectorize

"""csv_file_path = '/Users/namgyulee/Personal_Project/News_Article_Classification/Data/news_data.csv'

df = pd.read_csv(csv_file_path)

df['Description'] = df['Description'].fillna('')

title = df['Title']
description = df['Description']
author = df['Author'].astype(str)

# Tokenize and vectorize Train set
sentence_vectorizer = sentence_vectorize(title, description, author)
sentence_vectorizer.train_model()
train_sentence_vectors = sentence_vectorizer.get_sentence_vectors()
sentence_vectorizer.save_vectors_to_csv('normalized.csv')"""

new_df = pd.read_csv("/Users/namgyulee/Personal_Project/News_Article_Classification/Data/new_data.csv")
new_df['Description'] = new_df['Description'].fillna('')

title = new_df['Title']
description = new_df['Description']
author = new_df['Author'].astype(str)

# Tokenize and vectorize Train set
sentence_vectorizer = sentence_vectorize(title, description, author)
sentence_vectorizer.train_model()
train_sentence_vectors = sentence_vectorizer.get_sentence_vectors()
sentence_vectorizer.save_vectors_to_csv('/Users/namgyulee/Personal_Project/News_Article_Classification/Data/new_normalized.csv')

# Test Code before put functions in the class
"""
# Check wheather tokenized
for i in range(min(5, len(tokenized_description))):
    print( tokenized_description[i])

# Check wheather POS tagging
for i in range(min(5, len(posTag_description))):
    print(f'Text {i + 1} pos tag:', posTag_description[i])

# Check embedding well
word = 'northwestern'
word_vector = title_model.wv[pos_tag(tokenize_text(word))]
print(f'Vector representation of "{word}" : {word_vector}\n')
word_ava_vec = np.mean(word_vector)
print(word_ava_vec)
"""