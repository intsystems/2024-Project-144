from Popularity_Predictor import PopularityRecommender
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, ratings_train, items_df=None):
        self.items_df = items_df
        self.NUMBER_OF_FACTORS_MF = 30
        self.cf_predictions_df = self._calc_cf_predictions(ratings_train)
        self.PopularityRecommender = PopularityRecommender(ratings_train)

    def _calc_cf_predictions(self, ratings_train):

        users_items_pivot_matrix_df = ratings_train.pivot(index='user_id',
                                                          columns='movie_id',
                                                          values='rating').fillna(0)
        users_items_pivot_matrix = users_items_pivot_matrix_df.values
        users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)

        users_ids = list(users_items_pivot_matrix_df.index)

        #The number of factors to factor the user-item matrix.

        #Performs matrix factorization of the original user item matrix
        U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k=self.NUMBER_OF_FACTORS_MF)

        all_user_predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)

        all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (
                all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

        cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns=users_items_pivot_matrix_df.columns,
                                   index=users_ids).transpose()
        return cf_preds_df

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        if user_id in self.cf_predictions_df.columns.values:
            sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                .reset_index().rename(columns={user_id: 'rating'})

            # Recommend the highest predicted rating movies that the user hasn't seen yet.
            recommendations_df = sorted_user_predictions[~sorted_user_predictions['movie_id'].isin(items_to_ignore)] \
                .sort_values('rating', ascending=False) \
                .head(topn)

            # if verbose:
        #     if self.items_df is None:
        #         raise Exception('"items_df" is required in verbose mode')
        #
        #     recommendations_df = recommendations_df.merge(self.items_df, how = 'left',
        #                                                   left_on = 'movie_id',
        #                                                   right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]

        else:
            recommendations_df = self.PopularityRecommender.recommend_items(user_id, items_to_ignore=items_to_ignore,
                                                                            topn=topn, verbose=verbose)

        return recommendations_df


