# Movie Recommender System

The Movie Recommender System utilizes collaborative filtering to recommend movies to users based on their preferences. This system is developed with an approach of matrix factorization using tensorflow. This is the link of the app https://movienights.streamlit.app/

## Table of Contents
- [Overview](#Overview)
- [File Structure](#file-structure)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the System](#running-the-system)
- [MovieRecommender Class](#movierecommender-class)
  - [Methods](#methods)
- [Streamlit App](#streamlit-app)
- [Usage Notes](#usage-notes)


## Overview
The Movie Recommender System employs an embeddings-based approach coupled with matrix factorization to derive latent factors and predict user ratings for movies. This technique involves reducing dimensionality and creating vectors that encapsulate user and movie features.

### Key Concepts

- **Embeddings:** Embeddings are low-dimensional representations of high-dimensional data. In this context, they represent users and movies as vectors in a reduced space.
  
- **Matrix Factorization:** Matrix factorization is a method that decomposes a matrix into multiple matrices, which aids in identifying latent factors or hidden patterns within the data.
  
- **Latent Factors:** Latent factors are abstract and hidden features derived from the matrix factorization process. These factors could represent various characteristics like genre preferences, time impacts, influential actors, or other underlying features that affect user movie preferences.

### Idea

The primary concept revolves around predicting user reactions to movies based on similar user reactions. By creating embeddings for users and movies, we reduce the dimensionality and generate vectors that are used in deep learning algorithms. Matrix factorization is done by splitting the user movie matrix of ratings into user and movie matrix. The dimension in which it is broken is known as letent dimension.
- **For Example:** Let us suppose we have a matrix of m*n . It gets broken into m*d and d*n matrixes.



## File Structure

- **data/**: Directory containing CSV files with movie ratings, user information, and movie details.
- **model/**: Contains the MovieRecommender class and deployment script.
- **requirements.txt**: File specifying required Python libraries and versions.

## Usage

### Installation

1. Clone this repository:
    ```bash
    https://github.com/AkshatSan/MovieRecomm.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the System

1. Set up your data files in the `data/` directory.

2. Run the Movie Recommender System:
    ```bash
    cd model/
    python deploy.py
    ```

3. Access the Streamlit app in your web browser.
    ```bash
    https://movienights.streamlit.app/
    ```

## MovieRecommender Class

The `MovieRecommender` class contains methods for managing movie recommendations:

### Methods

- `__init__(base_path)`: Initializes the MovieRecommender object.
- `load_data()`: Loads required CSV files for movie ratings, user information, and movie details.
- `predict_rating(user_id, movie_id)`: Predicts the rating for a given user and movie.
- `recommend(TEST_USER)`: Provides top movie recommendations for a user.
- `display_recommendations()`: Displays movie recommendations in the Streamlit app.

## Streamlit App

The Streamlit app provides a user-friendly interface to select a user and view movie recommendations.

## Usage Notes

- Ensure proper setup of the data files in the `data/` directory before running the application.
- The Streamlit app allows users to select a user ID and view top movie recommendations.


