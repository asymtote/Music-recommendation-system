from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)

# Spotify API credentials
CLIENT_ID = "f334f87680604aa18a54ea468bdde0d7"
CLIENT_SECRET = "78d6400c39e04f9ebf4c30e1d082f6f3"

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load data and similarity matrix
music_df = pickle.load(open('df.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Spotify album cover retrieval and track URL
def get_album_cover_and_url(song_name, artist_name):
    query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=query, type="track", limit=1)

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        return track["album"]["images"][0]["url"], track["external_urls"]["spotify"]
    
    results = sp.search(q=f"track:{song_name}", type="track", limit=1)
    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        return track["album"]["images"][0]["url"], track["external_urls"]["spotify"]
    
    return "https://i.postimg.cc/0QNxYz4V/social.png", None

# Home route
@app.route('/')
def index():
    return render_template(
        'index.html', 
        songs=music_df['title'].tolist(),
        heroes=music_df['hero'].unique()
    )

# Song-based recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    selected_song = request.form['song']

    # Ensure consistency in song names by stripping spaces and converting to lowercase
    selected_song = selected_song.strip().lower()

    # Check if the song exists in the dataset
    music_df['title'] = music_df['title'].str.strip().str.lower()  # Ensure consistency across song names
    if selected_song not in music_df['title'].values:
        return render_template('index.html', error="Song not found in the database.", songs=music_df['title'].tolist())

    song_index = music_df[music_df['title'] == selected_song].index[0]
    distances = sorted(list(enumerate(similarity[song_index])), reverse=True, key=lambda x: x[1])

    recommended_songs = []
    recommended_posters = []
    recommended_links = []

    for i in distances[1:6]:  # Get top 5 recommendations
        song_title = music_df.iloc[i[0]]['title']
        artist = music_df.iloc[i[0]]['artist']
        poster, spotify_url = get_album_cover_and_url(song_title, artist)
        recommended_songs.append(song_title)
        recommended_posters.append(poster)
        recommended_links.append(spotify_url)

    return render_template(
        'recommend.html', 
        recommendations=recommended_songs, 
        posters=recommended_posters, 
        links=recommended_links, 
        zip=zip
    )

# Hero-based recommendation route
@app.route('/recommend_hero', methods=['POST'])
def recommend_hero():
    selected_hero = request.form['hero']
    hero_songs = music_df[music_df['hero'] == selected_hero]

    if hero_songs.empty:
        return render_template('recommend.html', recommendations=[], posters=[], links=[], zip=zip)
    
    recommended_songs = []
    recommended_posters = []
    recommended_links = []
    
    for _, row in hero_songs.iterrows():
        poster, spotify_url = get_album_cover_and_url(row['title'], row['artist'])
        recommended_songs.append(row['title'])
        recommended_posters.append(poster)
        recommended_links.append(spotify_url)
    
    return render_template(
        'recommend.html', 
        recommendations=recommended_songs[:5],  # Limit to 5 recommendations
        posters=recommended_posters[:5], 
        links=recommended_links[:5], 
        zip=zip
    )

# Year range-based recommendation route
@app.route('/recommend_year_range', methods=['POST'])
def recommend_year_range():
    year_range = request.form['year_range']
    start_year, end_year = map(int, year_range.split('-'))
    
    # Filter songs within the specified year range
    year_songs = music_df[(music_df['year'] >= start_year) & (music_df['year'] <= end_year)]

    if year_songs.empty:
        return render_template('recommend.html', recommendations=[], posters=[], links=[], zip=zip)

    # Cache Spotify results to optimize performance
    cache = {}
    recommended_songs, recommended_posters, recommended_links = [], [], []
    
    for _, row in year_songs.head(5).iterrows():  # Limit to 5 songs
        key = (row['title'], row['artist'])
        if key in cache:
            poster, link = cache[key]
        else:
            poster, link = get_album_cover_and_url(row['title'], row['artist'])
            cache[key] = (poster, link)
        
        recommended_songs.append(row['title'])
        recommended_posters.append(poster)
        recommended_links.append(link)

    return render_template(
        'recommend.html',
        recommendations=recommended_songs,
        posters=recommended_posters,
        links=recommended_links,
        zip=zip
    )

if __name__ == '__main__':
    app.run(debug=True)
                                                                            