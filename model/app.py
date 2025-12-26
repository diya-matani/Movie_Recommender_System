import streamlit as st
import pickle
import pandas as pd

# Custom CSS for UI - Deployment Fix v2
st.markdown("""
<style>
    .stApp {
        background-color: #141414;
        color: #ffffff;
    }
    h1 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #e50914; /* Netflix Red */
        font-weight: 800;
        text-align: center;
        margin-bottom: 30px;
    }
    .recommendation-card {
        background-color: #1f1f1f;
        border-left: 5px solid #e50914;
        border-radius: 5px;
        padding: 20px;
        margin-bottom: 15px;
        transition: transform 0.2s;
        display: flex;
        align-items: center;
    }
    .recommendation-card:hover {
        transform: translateX(10px);
        background-color: #2f2f2f;
    }
    .movie-rank {
        font-size: 24px;
        font-weight: bold;
        color: #666;
        margin-right: 20px;
        width: 30px;
    }
    .movie-title {
        color: #fff;
        font-size: 20px;
        font-weight: 500;
        margin: 0;
    }
    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton>button {
        background-color: #e50914;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #f40612;
        border-color: #f40612;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

import os
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cache models to prevent reloading on every interaction
@st.cache_resource
def load_models():
    # Define paths relative to this script file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    movie_dict_path = os.path.join(current_dir, 'movie_dict.pkl')
    similarity_path = os.path.join(current_dir, 'similarity.pkl')
    movies_csv_path = os.path.join(current_dir, 'tmdb_5000_movies.csv')
    credits_csv_path = os.path.join(current_dir, 'tmdb_5000_credits.csv')

    # check if pkl files exist
    if os.path.exists(movie_dict_path) and os.path.exists(similarity_path):
        movie_list = pickle.load(open(movie_dict_path,'rb'))
        similarity_matrix = pickle.load(open(similarity_path,'rb'))
        return pd.DataFrame(movie_list), similarity_matrix
    else:
        # If models are missing (e.g. on Streamlit Cloud), generate them!
        with st.spinner('Building model for the first time... (This takes a minute)'):
            if not os.path.exists(movies_csv_path) or not os.path.exists(credits_csv_path):
                st.error(f"Required CSV files not found at: {movies_csv_path}")
                return pd.DataFrame(), []

            movies = pd.read_csv(movies_csv_path)
            credits = pd.read_csv(credits_csv_path)
            
            movies = movies.merge(credits, on='title')
            movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
            
            def convert(obj):
                L = []
                try:
                    for i in ast.literal_eval(obj):
                        L.append(i['name'])
                except: pass
                return L
            
            def convert3(obj):
                L = []
                counter = 0
                try:
                    for i in ast.literal_eval(obj):
                        if counter != 3:
                            L.append(i['name'])
                            counter += 1
                        else: break
                except: pass
                return L
            
            def fetch_director(obj):
                L = []
                try:
                    for i in ast.literal_eval(obj):
                        if i['job'] == 'Director':
                            L.append(i['name'])
                            break
                except: pass
                return L

            movies.dropna(inplace=True)
            movies['genres'] = movies['genres'].apply(convert)
            movies['keywords'] = movies['keywords'].apply(convert)
            movies['cast'] = movies['cast'].apply(convert3)
            movies['crew'] = movies['crew'].apply(fetch_director)
            
            movies['overview'] = movies['overview'].apply(lambda x: x.split())
            movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
            
            movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
            
            new_df = movies[['movie_id', 'title', 'tags']]
            new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
            new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
            
            cv = CountVectorizer(max_features=5000, stop_words='english')
            vectors = cv.fit_transform(new_df['tags']).toarray()
            similarity = cosine_similarity(vectors)
            
            return new_df, similarity

movies, similarity = load_models()

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
        
    return recommended_movies

st.title('🎬 MovieFlix Recommender')

selected_movie_name = st.selectbox(
    'Search for a movie...',
    movies['title'].values
)

if st.button('Show Recommendations'):
    names = recommend(selected_movie_name)
    
    st.markdown("### Top Picks for You")
    
    # Text-only vertical list layout
    for i, name in enumerate(names):
        st.markdown(f"""
        <div class="recommendation-card">
            <div class="movie-rank">{i+1}</div>
            <div class="movie-title">{name}</div>
        </div>
        """, unsafe_allow_html=True)
