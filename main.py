import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import requests

dataset = pd.read_csv('anime-dataset.csv')
# print(dataset.columns)
# print(dataset.shape)

important_features = ["Genres", "Synopsis", "Type", "Aired", "Studios", "Source", "Rating"]

for feature in important_features:
    dataset[feature] = dataset[feature].replace('UNKNOWN', np.nan)
    dataset[feature] = dataset[feature].fillna("")

dataset['combined featrures'] = dataset["Genres"] + ' ' + dataset["Synopsis"] + ' ' + dataset["Type"] + ' ' + dataset["Aired"] + ' ' + dataset["Studios"] + ' ' +  dataset["Source"] + ' ' +dataset["Rating"]
# print(dataset["combined featrures"].head())
vectorizer = TfidfVectorizer()
vectorized_features = vectorizer.fit_transform(dataset["combined featrures"])
# print(vectorized_features.shape)

# similarity = cosine_similarity(vectorized_features)
# print(similarity.shape)
all_anime_jap = dataset['Name'].tolist()
all_anime_eng = dataset['English name'].tolist()

print(all_anime_jap)
print(all_anime_eng)