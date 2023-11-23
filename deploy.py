import pandas as pd
import numpy as np
import tensorflow
#import seaborn as sns
from tensorflow import keras 
import matplotlib
import streamlit as st

class MovieRecommender:
    """
    A class for movie recommendation using collaborative filtering.

    Attributes:
    - base_path (str): Base path to the data directory.
    - ratings (DataFrame): DataFrame containing user ratings.
    - users (DataFrame): DataFrame containing user details.
    - movies (DataFrame): DataFrame containing movie details.
    - new_model (keras.Model): Keras model for movie rating prediction.

    Methods:
    - load_data: Load required CSV files.
    - predict_rating: Predict the rating for a given user and movie.
    - recommend: Provide movie recommendations for a user.
    - display_recommendations: Display movie recommendations in the Streamlit app.

    """
    
    def __init__(self, base_path):
        """
        Initialize the MovieRecommender object.

        Args:
        - base_path (str): Base path to the data directory.
        """
        self.base_path = base_path
        self.load_data()

    def load_data(self):
        """
        Loading all the required csv
        """
        self.ratings = pd.read_csv('data/ratings.csv', sep='\t', encoding='latin-1', 
                                   usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
        self.users = pd.read_csv('data/users.csv', sep='\t', encoding='latin-1', 
                                 usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])
        self.movies = pd.read_csv('data/movies.csv', sep='\t', encoding='latin-1', 
                                  usecols=['movie_id', 'title', 'genres'])
        self.new_model = keras.models.load_model('data/newmodel.h5')

    def predict_rating(self, user_id, movie_id):
        """
        Predict the rating for a given user and movie.

        Args:
        - user_id (int): User ID for prediction.
        - movie_id (int): Movie ID for prediction.

        Returns:
        - float: Predicted rating for the user-movie pair.
        """
        return self.new_model.predict([np.array([user_id]), np.array([movie_id])], verbose=0)[0][0]

    def recommend(self, TEST_USER):

        """
        Provide movie recommendations for a user.

        Args:
        - TEST_USER (int): User ID for recommendation.

        Returns:
        - DataFrame: DataFrame containing top 10 movie recommendations for the user.
        """
        user_ratings = self.ratings[self.ratings['user_id'] == TEST_USER][['user_id', 'movie_id', 'rating']]
        recommendations = self.ratings[self.ratings['movie_id'].isin(user_ratings['movie_id']) == False][['movie_id']].drop_duplicates()
        recommendations=recommendations.sample(n=500,random_state=42)
        recommendations['prediction'] = recommendations.apply(lambda x: self.predict_rating(TEST_USER, x['movie_id']), axis=1)
        return recommendations.sort_values(by='prediction',
                                           ascending=False).merge(self.movies,
                                                                  on='movie_id',
                                                                  how='inner'
                                                                 ).head(10)

    def display_recommendations(self):


        """
        Streamlit function so as to give a frontend experience.
        """
        st.set_page_config(layout="wide", page_title="Movie Recommendation System", page_icon="🎬")
        st.markdown("""
        <style>
            .sidebar .sidebar-content {
                background-image: linear-gradient(to bottom, #f7f7f7, #f0f0f0);
            }
        </style>
        """, unsafe_allow_html=True)

        # Streamlit UI
        st.title('Movie Recommendation System')

        # Sidebar for user input
        user_ids = self.ratings['user_id'].unique()
        test_user_id = st.sidebar.selectbox('Select User ID:', user_ids)
        #test_user_id = st.sidebar.text_input('Enter User ID:', '1')  # Default user ID is set to 1
        test_user_id = int(test_user_id)

        if st.sidebar.button('Get Recommendations'):
            with st.spinner('Fetching Recommendations...'):
                recommended_movies = self.recommend(test_user_id)
                st.subheader(f"Top 20 Movie Recommendations for User ID {test_user_id}:")
                #st.dataframe(recommended_movies)
                recommended_movies = recommended_movies[['prediction', 'title', 'genres']]
                styled_recommendations = recommended_movies.style\
                .bar(subset=['prediction'], color='#FFA07A')\
                .background_gradient(cmap='Blues', subset=['prediction'])\
                .highlight_max(subset=['prediction'], color='lightgreen')\
                .highlight_min(subset=['prediction'], color='lightcoral')\
                .set_properties(**{'text-align': 'center'})
                
            
            st.dataframe(styled_recommendations)

if __name__ == "__main__":
    print("Execution started")
    base_path = '..'  # Replace this with your base path
    movie_recommender = MovieRecommender(base_path)
    movie_recommender.display_recommendations()
    print("Execution ended")
