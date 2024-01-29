import numpy as np
import pandas as pd
from function import get_API_key, gpt_Model
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from openai import OpenAI
import json

df = pd.read_csv("/Users/namgyulee/Personal_Project/News_Article_Classification/Data/news_data.csv")

title = json.dumps(df['Title'].tolist())
key_file = '/Users/namgyulee/Personal_Project/News_Article_Classification/api-key.txt'
api_key_instance = get_API_key(key_file, 5)
API_key = api_key_instance.get_api_key(5)

client = OpenAI(
    api_key = API_key
    )

gpt_model_instance = gpt_Model(client)

# Get embeddings for each chunk of 'Title', 'Description', and 'Author' columns
title_embeddings = gpt_model_instance.get_embeddings(title)