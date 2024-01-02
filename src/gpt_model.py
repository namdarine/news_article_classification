from sklearn.model_selection import train_test_split
from transformers import OpenAIGPTTokenizer, TFOpenAIGPTModel
import tensorflow as tf
import numpy as np
import pandas as pd

# Tokenize data using GPTTokenizer
df = pd.read_csv('/Users/namgyulee/Personal_Project/News_Article_Classification/Data/news_data.csv')

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
#df['Description'] = df['Description'].fillna('')
"""
title = df['Title']
des = df['Description']
#author = df['Author'].astype(str)
author = df['Author']

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
token_title = tokenizer(title, return_tensors="pt", padding=True, truncation=True)
token_des = tokenizer(des, return_tensors="pt", padding=True, truncation=True)
token_author = tokenizer(author, return_tensors="pt", padding=True, truncation=True)


# Train GPT using Transformers module


>>> model = TFOpenAIGPTModel.from_pretrained('openai-gpt')

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs[0]
"""