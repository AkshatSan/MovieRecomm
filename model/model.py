from keras.layers import Input, Reshape, Dot, Embedding
from keras.models import Model

class MFCFModel:
    def __init__(self, n_users, m_items, latent_factors):
        """
    The model expects
    n_users:-No of users
    m_items:- No of items
    latent_factors:- No of dimensions the data has to be structures
    Consider latent factors as features which gets picked up and are used to split .
    
    """

        self.n_users = n_users
        self.m_items = m_items
        self.latent_factors = latent_factors
        
    def create_model(self):

        """
        This function does the main step that is matrix factorization.
        It converts the user-movie 2d matrix of ratings into user and movie .
        The dimension of user and movie will be decided upon the number of latent_features.
        We are then doing dot product so this is basically application of low rank matrix factorization
        
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
