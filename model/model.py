from keras.layers import Input, Reshape, Dot, Embedding
from keras.models import Model

class MFCFModel:
    def __init__(self, n_users, m_items, latent_factors):
        """
        Initializes the Matrix Factorization Collaborative Filtering (MFCF) model.

        Args:
        - n_users (int): Number of users in the dataset.
        - m_items (int): Number of items (movies) in the dataset.
        - latent_factors (int): Number of dimensions for latent factors.

        Consider latent factors as features which get picked up and are used to split.
        """

        self.n_users = n_users
        self.m_items = m_items
        self.latent_factors = latent_factors
        
    def create_model(self):
        """
        Creates the Matrix Factorization Collaborative Filtering (MFCF) model.

        This function performs matrix factorization, converting the user-movie 2D matrix of ratings into
        user and movie dimensions based on the number of latent features.
        It employs a low-rank matrix factorization approach by applying dot product operations.

        Returns:
        - Model: Compiled MFCF model.
        """

        user = Input(shape=(1,))
        P = Embedding(self.n_users, self.latent_factors, input_length=1, name='user-embed')(user)
        P = Reshape((self.latent_factors,), name="user-reshape")(P)

        movie = Input(shape=(1,))
        Q = Embedding(self.m_items, self.latent_factors, input_length=1, name='movie-embed')(movie)
        Q = Reshape((self.latent_factors,), name="movie-reshape")(Q)

        P_dot_Q = Dot(axes=1, name="dot_product")([P, Q])
        model = Model(inputs=[user, movie], outputs=P_dot_Q, name="output")

        return model
