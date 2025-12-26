import streamlit as st
import pickle
import pandas as pd

# Custom CSS for UI
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

# Cache models to prevent reloading on every interaction
@st.cache_resource
def load_models():
    movie_list = pickle.load(open('movie_dict.pkl','rb'))
    similarity_matrix = pickle.load(open('similarity.pkl','rb'))
    return pd.DataFrame(movie_list), similarity_matrix

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
