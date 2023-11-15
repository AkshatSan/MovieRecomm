import pandas as pd
import numpy as np
from tensorflow import keras 
import streamlit as st

base_path = '..'
ratings = pd.read_csv(base_path + '//data//ratings.csv', sep='\t', encoding='latin-1', 
                      usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
users = pd.read_csv(base_path + '/data/users.csv', sep='\t', encoding='latin-1', 
                    usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])
movies = pd.read_csv(base_path + '/data/movies.csv', sep='\t', encoding='latin-1', 
                     usecols=['movie_id', 'title', 'genres'])
new_model=keras.models.load_model(base_path + '//data//newmodel.h5')

def predict_rating(model, user_id, movie_id):
    return model.predict([np.array([user_id]), np.array([movie_id])])[0][0]

def recommend(TEST_USER):
    user_ratings = ratings[ratings['user_id'] == TEST_USER][['user_id', 'movie_id', 'rating']]
    recommendations = ratings[ratings['movie_id'].isin(user_ratings['movie_id']) == False][['movie_id']].drop_duplicates()
    recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(new_model, TEST_USER, x['movie_id']), axis=1)
    return recommendations.sort_values(by='prediction',
                          ascending=False).merge(movies,
                                                 on='movie_id',
                                                 how='inner'
                                                 ).head(20)

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
test_user_id = st.sidebar.text_input('Enter User ID:', '1')  # Default user ID is set to 1
test_user_id = int(test_user_id)

if st.sidebar.button('Get Recommendations'):
    with st.spinner('Fetching Recommendations...'):
        recommended_movies = recommend(test_user_id)
        st.subheader(f"Top 20 Movie Recommendations for User ID {test_user_id}:")
        st.dataframe(recommended_movies)