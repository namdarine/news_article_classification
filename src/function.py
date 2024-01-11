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
    def __init__(self, head, model, api_key):
        self.head = head
        self.api_key = api_key
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = model
    """    
    def tokenizing_text(self):
        if isinstance(self.head, str):
            # If the input is a single string
            token = self.tokenizer(self.head, return_tensors="pt", padding=True, truncation=True)
        elif isinstance(self.head, list):
            # If the input is a list of strings
            token = self.tokenizer(self.head, return_tensors="pt", padding=True, truncation=True, is_split_into_words=True)
        elif isinstance(self.head, list) and all(isinstance(item, list) for item in self.head):
            # If the input is a list of lists of strings
            token = self.tokenizer(self.head, return_tensors="pt", padding=True, truncation=True, is_split_into_words=True)
        else:
            raise ValueError("Invalid input type. Must be a string, list of strings, or list of lists of strings.")
        
        return token
    """
    def get_embedding(self):
        client = OpenAI(
              api_key=os.environ[f'self.api_key'],  # this is also the default, it can be omitted
            )
       
        text = self.head.replace("\n", " ")
        return client.embeddings.create(input = [text], model=self.model).data[0].embedding

"""
    def embedding_text(self):
        token_head = self.tokenizing_text()
    
        if isinstance(self.head, list):
            input_text = " ".join(self.head)
        else:
            input_text = self.head
    
        client = OpenAI(
              api_key=os.environ[self.api_key],  # this is also the default, it can be omitted
            )
        
        response = openai.Completion.create(
            engine="text-embedding-ada-002",
            prompt=input_text,
            temperature=0.5
        )
        embedding = response['choices'][0]['model_output']['embeddings']
    
        return embedding
    
"""
