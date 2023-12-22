import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import fasttext


class sentence_vectorize:
    def __init__(self, title, description):
        self.title = title
        self.description = description
        self.model = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.word_vectors = None
    
    def tokenize_text(self, text):
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens
    
    def train_model(self):
        all_tokenized = [self.tokenize_text(text) for text in self.title + self.description]
        all_tokenized_text = [" ".join(tokens) for tokens in all_tokenized]
        
        # Calculate TF-IDF
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(all_tokenized_text)
        
        # Train the model
        self.model = Word2Vec(sentences=all_tokenized, vector_size=100, window=5, min_count=1, workers=4)

        self.word_vectors = {word: self.model.wv[word] for word in self.model.wv.index_to_key}
        
    def get_sentence_vectors(self):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first")
        
        # Get sentence vectors each sentence
        sentence_vectors = []
        
        for tokens in [self.tokenize_text(text) for text in self.title + self.description]:
            sentence_vectors.append([])
        
            for word in tokens:
                if word in self.word_vectors and word in self.vectorizer.vocabulary_:
                    index_in_tfidf_matrix = tokens.index(word)
                    index_in_vocab = self.vectorizer.vocabulary_[word]
                    tfidf_value = self.tfidf_matrix[index_in_tfidf_matrix, index_in_vocab]
                    vector = tfidf_value * self.word_vectors[word]  # Use TF-IDF value as weight
                    sentence_vectors[-1].append(vector)
                    
            if sentence_vectors[-1]:
                sentence_vectors[-1] = np.mean(sentence_vectors[-1], axis=0)
        
        return sentence_vectors[0] + sentence_vectors[1]
    
    def save_vectors_to_csv(self, output_csv):
        sentence_vectors = self.get_sentence_vectors()
        
        df_output = pd.DataFrame({
            'Original Title Text': list(self.title),
            'Title Sentence Vector': sentence_vectors[0],
            'Original Description Text': list(self.description),
            'Description Sentence Vector': sentence_vectors[1]
            })
        df_output.to_csv(output_csv, index = True)


csv_file_path = '/Users/namgyulee/Personal_Project/News_Article_Classification/news_data.csv'

df = pd.read_csv(csv_file_path)

df['Description'] = df['Description'].fillna('')

title = df['Title']
description = df['Description']


train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# Split Title and Description
# They have same articles
title_train_set = train_set['Title']
title_test_set = test_set['Title']
des_train_set = train_set['Description']
des_test_set = test_set['Description']

sentence_vectorizer = sentence_vectorize(title_train_set, des_train_set)
sentence_vectorizer.train_model()
train_sentence_vectors = sentence_vectorizer.get_sentence_vectors()
sentence_vectorizer.save_vectors_to_csv('train_sentence_vectors.csv')


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


