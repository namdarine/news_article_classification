import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

csv_file_path = '/Users/namgyulee/Personal_Project/news_data.csv'

df = pd.read_csv(csv_file_path)

df['Description'] = df['Description'].fillna('')

title = df['Title']
description = df['Description']
print(type(description))

# Function for Tokenize, stopwords, stemming, and lemmatization
def tokenize_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Tokenize Title and Description
tokenized_title = [tokenize_text(text) for text in title]
tokenized_description = [tokenize_text(text) for text in description]

# Part-of-Speech Tagging 
posTag_title = [pos_tag(token) for token in tokenized_title]
posTag_description = [pos_tag(token) for token in tokenized_description]

# Word vector embedding using Word2Vec
title_model = Word2Vec(sentences=posTag_title, vector_size=100, window=5, min_count=1, workers=4)
description_model = Word2Vec(sentences=posTag_description, vector_size=100, window=5, min_count=1, workers=4)

"""
# Check embedding well
word = 'copyright'
word_vector = title_model.wv[pos_tag(tokenize_text(word))]
print(f'Vector representation of "{word}" : {word_vector}')


# Check wheather POS tagging
for i in range(min(5, len(posTag_description))):
    print(f'Text {i + 1} pos tag:', posTag_description[i])

# Check wheather tokenized
for i in range(min(5, len(tokenized_description))):
    print(f'Text {i + 1} tokens:', tokenized_description[i])
"""

#train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

#print(f"size of a training set: {len(train_set)}")
#print(f"size of a test set: {len(test_set)}")