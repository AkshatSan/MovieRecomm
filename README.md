# Movie Recommender System

The Movie Recommender System utilizes collaborative filtering to recommend movies to users based on their preferences. This system is developed using Python, leveraging Pandas, NumPy, TensorFlow, and Streamlit libraries for data processing, modeling, and user interface. This is the link of the app https://movienights.streamlit.app/

## Table of Contents

- [File Structure](#file-structure)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the System](#running-the-system)
- [MovieRecommender Class](#movierecommender-class)
  - [Methods](#methods)
- [Streamlit App](#streamlit-app)
- [Usage Notes](#usage-notes)

## File Structure

- **data/**: Directory containing CSV files with movie ratings, user information, and movie details.
- **model/**: Contains the MovieRecommender class and deployment script.
- **requirements.txt**: File specifying required Python libraries and versions.

## Usage

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your_username/movie-recommender.git
    cd movie-recommender
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


