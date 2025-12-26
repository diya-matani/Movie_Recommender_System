# Movie Recommender System

A content-based movie recommender system using the TMDB 5000 Movie Dataset.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Models:**
    - Open `notebook.ipynb` in Jupyter Notebook or VS Code.
    - Run all cells to process the data and generate `movie_dict.pkl` and `similarity.pkl`.

3.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## Files

-   `app.py`: The main Streamlit application.
-   `notebook.ipynb`: Jupyter notebook for data processing and model creation.
-   `requirements.txt`: List of Python dependencies.
-   `tmdb_5000_movies.csv`: Dataset containing movie details.
-   `tmdb_5000_credits.csv`: Dataset containing cast and crew details.
