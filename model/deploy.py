import pandas as pd
import numpy as np
from tensorflow import keras 
import streamlit as st

class MovieRecommender:
    def __init__(self, base_path):
        self.base_path = base_path
        self.load_data()

    def load_data(self):
        """
        Loading all the required csv
        """
        self.ratings = pd.read_csv(self.base_path + '//data//ratings.csv', sep='\t', encoding='latin-1', 
                                   usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
        self.users = pd.read_csv(self.base_path + '/data/users.csv', sep='\t', encoding='latin-1', 
                                 usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])
        self.movies = pd.read_csv(self.base_path + '/data/movies.csv', sep='\t', encoding='latin-1', 
                                  usecols=['movie_id', 'title', 'genres'])
        self.new_model = keras.models.load_model(self.base_path + '//data//newmodel.h5')

    def predict_rating(self, user_id, movie_id):
        """
        user_id:- the user id that needs the recommendation
        movie_id:- The movie_id which will be rated

        The function accepts the user_id and movie_id and predicts the rating.
        
        """
        return self.new_model.predict([np.array([user_id]), np.array([movie_id])], verbose=0)[0][0]

    def recommend(self, TEST_USER):

        """
        TEST_USER:- the user_id for reccommendation

        The function accepts the user_id and gives the top 20 recommendation to the user
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
    base_path = '..'  # Replace this with your base path
    movie_recommender = MovieRecommender(base_path)
    movie_recommender.display_recommendations()
