import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import OpenAIGPTTokenizer
from openai import OpenAI
import os

class get_API_key:
    def __init__(self, filename, n):
        self.filename = filename
        self.n = n
        
    def get_api_key(self, line_number):
        with open(self.filename, 'r') as file:
            lines = file.readlines()
            if 1 <= line_number <= len(lines):
                api_key = lines[line_number - 1].strip()
                return api_key
            else:
                print(f"Error: Line {line_number} does not exist in the file.")
                return None

class sentence_vectorize:
    def __init__(self, title, description, author):
        self.title = title
        self.description = description
        self.author = author
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
        all_tokenized = [self.tokenize_text(text) for text in self.title + self.description + self.author]
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
        
        # Get sentence vectors for each sentence
        sentence_vectors = []
    
        for tokens in [self.tokenize_text(text) for text in self.title + self.description + self.author]:
            sentence_vector = np.zeros(self.model.vector_size)  # Initialize a vector of zeros
    
            for word in tokens:
                if word in self.word_vectors and word in self.vectorizer.vocabulary_:
                    index_in_vocab = self.vectorizer.vocabulary_[word]
                    tfidf_value = self.vectorizer.transform([word]).toarray()[0, index_in_vocab]
                    vector = tfidf_value * self.word_vectors[word]  # Use TF-IDF value as weight
                    sentence_vector += vector
    
            sentence_vectors.append(sentence_vector)
    
        return sentence_vectors


    def save_vectors_to_csv(self, output_csv):
        sentence_vectors = self.get_sentence_vectors()
    
        df_output = pd.DataFrame({
            'Original Title Text': list(self.title),
            'Title Sentence Vector': [vector[0] for vector in sentence_vectors],
            'Original Description Text': list(self.description),
            'Description Sentence Vector': [vector[1] for vector in sentence_vectors],
            'Original Author Text': list(self.author),
            'Author Sentence Vector': [vector[2] for vector in sentence_vectors]
        })
        df_output.to_csv(output_csv, index=True) 

class gpt_Model:
    def __init__(self, client, model="text-embedding-ada-002"):
        self.client = client
        self.model = model

    def get_embedding(self, text: str):
        response = self.client.embeddings.create(input=[text], model=self.model)
        embedding = response.data[0].embedding
        return embedding

    def get_embeddings(self, text_list):
        embeddings = []
        for text_chunk in text_list:
            embedding = self.get_embedding(text_chunk)
            embeddings.append(embedding)
        return embeddings
    
    
    
    