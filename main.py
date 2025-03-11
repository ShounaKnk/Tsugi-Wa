import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import os
import streamlit as st


st.title("hello World")
# Load dataset
dataset = pd.read_csv('anime-dataset.csv')

# List of important features

# Reset index and create AnimeIndex column for consistent reference
dataset = dataset.reset_index(drop=True)
dataset['AnimeIndex'] = dataset.index

# Check if similarity matrix exists, else compute it
similarity_matrix = 'cosine_similarity_matrix.npy'
if os.path.exists(similarity_matrix):
    similarity = np.load(similarity_matrix)
    print('Loaded saved similarity matrix')
else:
    important_features = ["Genres", "Synopsis", 'Source', 'Studios', 'Aired', 'Rating']

    # Fill missing values
    for feature in important_features:
        dataset[feature] = dataset[feature].replace('UNKNOWN', 'not_specified').fillna('not_specified')

    # Combine important features for similarity calculation
        dataset['combined_features'] = (
        (dataset["Genres"] + " ")*3 + 
        (dataset["Rating"]+" ")*2+
        (dataset["Source"] + " ")*2 + 
        (dataset["Studios"]+ " ")*2 + 
        (dataset["Synopsis"] + " ")*1 + 
        (dataset["Aired"])*1
    )
    vectorizer = TfidfVectorizer()
    vectorized_features = vectorizer.fit_transform(dataset["combined_features"])
    similarity = cosine_similarity(vectorized_features)
    np.save(similarity_matrix, similarity)
    print('Computed and saved the similarity matrix')

# Remove duplicate anime entries based on EnglishName
dataset = dataset.drop_duplicates(subset=['EnglishName'], keep='first')

# Get all anime names
all_anime_jap = dataset['JapaneseName'].tolist()
all_anime_eng = dataset['EnglishName'].tolist()

# Input anime name
anime_name = input('Enter your favourite anime: ')

def find_best_match(anime_name, anime_list):
    close_matches = difflib.get_close_matches(anime_name, anime_list)
    return close_matches[0] if close_matches else None

# Try finding the closest match in English first, then Japanese
closest_match = find_best_match(anime_name, all_anime_eng)
jap = False
if closest_match is None:
    jap = True
    closest_match = find_best_match(anime_name, all_anime_jap)

# If no match found, exit
if closest_match is None:
    print("No matching anime found.")
    exit()
else:
    print("anime similar to ", closest_match, ": ")

# Get the index of the closest matching anime
if jap:
    anime_index = dataset.loc[dataset.JapaneseName == closest_match, 'AnimeIndex'].values[0]
else:
    anime_index = dataset.loc[dataset.EnglishName == closest_match, 'AnimeIndex'].values[0]

# Compute similarity scores
similarity_score = list(enumerate(similarity[anime_index]))
sorted_similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)

# Function to get anime name by index
def get_anime_name(index, language):
    result = dataset.loc[dataset.AnimeIndex == index, language].values
    return result[0] if result.size > 0 else None

utter_trash = ["Love Live!", "BanG Dream!", "The iDOLM@STER"]
def isTrash(anime):
    for trash in utter_trash:
        if trash.lower() in anime.lower():
            return True

# Store unique recommendations to avoid duplicates
recommended_anime = []
i = 1
print("\nRecommended Anime:")
for index, score in sorted_similarity_score[1:50]:  # Check more entries to ensure unique suggestions
    if not jap:
        anime_from_index = get_anime_name(index, 'EnglishName')
        if anime_from_index == 'UNKNOWN' or anime_from_index is None:
            anime_from_index = get_anime_name(index, 'JapaneseName')
    else:
        anime_from_index = get_anime_name(index, 'JapaneseName')
    popularity = dataset[dataset.AnimeIndex == index]["Popularity"].values
    score = dataset[dataset.AnimeIndex == index]["Score"].values

    if anime_from_index and anime_from_index not in recommended_anime:
        if not closest_match in anime_from_index:
            if isTrash(anime_from_index):
                continue
            if popularity.size > 0 and score.size>0:
                popularity = float(popularity[0])
                score = score[0]
                if popularity > 90 and 7.0 <= score <=10.0:
                    recommended_anime.append((anime_from_index, score))
            i += 1
recommended_anime.sort(key=lambda x:x[1], reverse=True)
i = 1
for recomendation, score in recommended_anime:
    print(f"{i}. {recomendation} (Score {score})")
    i+=1


# If no recommendations found
if i == 1:
    print("No similar anime found.")

