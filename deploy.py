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
        self.preprocess()

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
        print(len(self.ratings))
    def preprocess(self):
        user_rating=self.users.merge(self.ratings,how='inner',on='user_id')
        self.df=user_rating.merge(self.movies,how='inner',on='movie_id')
        self.df.drop(columns=['user_emb_id','occ_desc','movie_emb_id','zipcode'],axis=1,inplace=True)
        #self.df.rename(columns={'user_emb_id':'user_id',
                        #'movie_emb_id':'movie_id'},inplace=True)
        genres = self.df['genres'].str.get_dummies(sep='|')

        # Combining the original DataFrame with the one-hot encoded genres
        self.df = pd.concat([self.df, genres], axis=1).drop('genres', axis=1)


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

    def recommend(self, TEST_USER,um_recommendations):

        """
        Provide movie recommendations for a user.

        Args:
        - TEST_USER (int): User ID for recommendation.

        Returns:
        - DataFrame: DataFrame containing top 10 movie recommendations for the user.
        """
        print(len(self.ratings))
        user_ratings = self.ratings[self.ratings['user_id'] == TEST_USER][['user_id', 'movie_id', 'rating']]
        recommendations = self.ratings[self.ratings['movie_id'].isin(user_ratings['movie_id']) == False][['movie_id']].drop_duplicates()
        if len(recommendations)>500:
            recommendations=recommendations.sample(n=300,random_state=42)
        recommendations['prediction'] = recommendations.apply(lambda x: self.predict_rating(TEST_USER, x['movie_id']), axis=1)
        return recommendations.sort_values(by='prediction',
                                           ascending=False).merge(self.movies,
                                                                  on='movie_id',
                                                                  how='inner'
                                                                 ).head(um_recommendations)

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
        um_recommendations = st.number_input('Enter the number of recommendations', min_value=1, max_value=100, value=10)
        um_recommendations=int(um_recommendations)
        #test_user_id = st.sidebar.text_input('Enter User ID:', '1')  # Default user ID is set to 1
        test_user_id = int(test_user_id)
        st.write('Select genres to filter the movies')

        # Get unique genres from the DataFrame columns (assuming genres start from column index 6)
        all_genres = self.df.columns[6:]

        # Multi-select widget for user to choose genres
        selected_genres = st.multiselect('Select genres', all_genres)

        # Create a filter based on selected genres
        genre_filter = self.df[selected_genres].any(axis=1)
        filtered_df = self.df[genre_filter]
        

        self.ratings=filtered_df[['user_id', 'movie_id', 'rating']].copy()

        if st.sidebar.button('Get Recommendations'):
            with st.spinner('Fetching Recommendations...'):
                recommended_movies = self.recommend(test_user_id,um_recommendations)
                st.subheader(f"Top {um_recommendations} Movie Recommendations for User ID {test_user_id}:")
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
