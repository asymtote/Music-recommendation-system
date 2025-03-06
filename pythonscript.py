# %%
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import random

df = pd.read_csv("cleaned_songs.csv")


df.info()

columns_to_drop = ['id', 'image', 'url', 'duration']
columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df = df.drop(columns_to_drop, axis=1)

genres = ['Pop', 'Rock', 'Classical', 'Jazz', 'Hip-hop', 'Country', 'Electronic', 'Blues']
df['genre'] = [random.choice(genres) for _ in range(len(df))]  # Random genres for demonstration


nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenization(txt):
    txt = str(txt).lower()  # Convert text to lowercase
    tokens = nltk.word_tokenize(txt)  # Tokenize the text
    stemming = [stemmer.stem(w) for w in tokens]  # Apply stemming
    return " ".join(stemming)

# Combine 'title', 'album', 'artist', 'hero', and 'genre' into one text column
df['text'] = df['title'] + ' ' + df['album'] + ' ' + df['artist'] + ' ' + df['hero'] + ' ' + df['genre']
df['text'] = df['text'].apply(lambda x: tokenization(x))


# Initialize TF-IDF Vectorizer and transform the text column
tfidvector = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidvector.fit_transform(df['text'])

# Calculate the cosine similarity matrix
similarity = cosine_similarity(matrix)

# Save the similarity matrix and the dataframe for later use
pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.dump(df, open('df.pkl', 'wb'))

def genre_based_recommendation(selected_genre):
    # Filter the DataFrame based on the user's selected genre
    genre_df = df[df['genre'] == selected_genre]

    if genre_df.empty:
        return f"No songs found for the genre: {selected_genre}"

    # Recompute TF-IDF and similarity matrix for the filtered dataset
    genre_tfidvector = TfidfVectorizer(analyzer='word', stop_words='english')
    genre_matrix = genre_tfidvector.fit_transform(genre_df['text'])
    genre_similarity = cosine_similarity(genre_matrix)

    # Recommend the top songs in the selected genre
    top_recommendations = genre_df.head(10)['title'].tolist()  # Get top 10 songs
    return top_recommendations



def hero_based_recommendation(selected_hero):
    # Filter songs associated with the selected hero
    hero_df = df[df['hero'] == selected_hero]
    if hero_df.empty:
        return f"No songs found for the hero: {selected_hero}"

    # Recommend the top songs
    top_recommendations = hero_df.head(10)['title'].tolist()  # Top 10 songs
    return top_recommendations



def song_based_recommendation(selected_song):
    # Get the genre of the selected song
    selected_song_data = df[df['title'] == selected_song]
    if selected_song_data.empty:
        return [], []

    selected_genre = selected_song_data.iloc[0]['genre']
    # Recommend songs in the same genre
    genre_recommendations = genre_based_recommendation(selected_genre)

    return genre_recommendations


# Example usage for genre-based recommendation
selected_genre = "Rock"  # Replace with the user's input
genre_recommendations = genre_based_recommendation(selected_genre)
print(f"Recommendations for genre {selected_genre}: {genre_recommendations}")



# Example usage:
selected_song = "Ola Olaala Ala"  # Replace with the user's input
recommendations = song_based_recommendation(selected_song)
print(f"Recommendations based on the song {selected_song}:")
print(recommendations)


# Example usage for hero-based recommendation
selected_hero = "Vishwak Sen"  # Replace with the user's input
hero_recommendations = hero_based_recommendation(selected_hero)
print(f"Recommendations for hero {selected_hero}: {hero_recommendations}")


