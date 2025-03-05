import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import os

dataset = pd.read_csv('anime-dataset.csv')

important_features = ["Genres", "Synopsis", "Type", "Aired", "Studios", "Source", "Rating"]

for feature in important_features:
    dataset[feature] = dataset[feature].replace('UNKNOWN', np.nan).fillna("")

dataset['combined_features'] =dataset["Genres"] + ' ' + dataset["Synopsis"] + ' ' + dataset["Type"] + ' ' +dataset["Aired"] + ' ' + dataset["Studios"] + ' ' + dataset["Source"] + ' ' +dataset["Rating"]

similarity_matrix = 'cosine_similarity_matrix.npy'
if os.path.exists(similarity_matrix):
    similarity = np.load(similarity_matrix)
    print('loaded saved similarity matrix')
else:
    vectorizer = TfidfVectorizer()
    vectorized_features = vectorizer.fit_transform(dataset["combined_features"])
    similarity = cosine_similarity(vectorized_features)
    np.save(similarity_matrix, similarity)
    print('computed and saved the similarity matrix')

all_anime_jap = dataset['JapaneseName'].tolist()
all_anime_eng = dataset['EnglishName'].tolist()

anime_name = input('Enter your favourite anime: ')

def find_in_list(anime_name, all_anime_lang):
    close_match = difflib.get_close_matches(anime_name, all_anime_lang)
    return close_match if close_match else -1

find_close_matches = find_in_list(anime_name, all_anime_eng)
jap = 0

if find_close_matches == -1:
    jap = 1
    find_close_matches = find_in_list(anime_name, all_anime_jap)
    
closest_match = find_close_matches[0]

if jap:
    anime_index = dataset[dataset.JapaneseName == closest_match]['AnimeIndex'].values[0]
else:
    anime_index = dataset[dataset.EnglishName == closest_match]['AnimeIndex'].values[0]

similarity_score = list(enumerate(similarity[anime_index]))
sorted_similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)

def get_anime_name(index, language):
    result = dataset[dataset.AnimeIndex == index][language].values
    return result[0] if result.size > 0 else None

i = 1
for anime in sorted_similarity_score[1: 10]:
    index = anime[0]
    if not jap:
        anime_from_index = get_anime_name(index, 'EnglishName')
        if anime_from_index == 'UNKNOWN' or anime_from_index is None:
            anime_from_index = get_anime_name(index, 'JapaneseName')
    else:
        anime_from_index = get_anime_name(index, 'JapaneseName')

    if anime_from_index:  # Check if not None or empty
        print(i, " ", anime_from_index)
        i += 1
    else:
        print('anime not found')