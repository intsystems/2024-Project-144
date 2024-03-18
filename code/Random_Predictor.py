class RandomRecommender:
    MODEL_NAME = 'Random Recommender'

    def __init__(self, ratings_train, items_df=None):
        self.movies = ratings_train[["movie_id"]].drop_duplicates()

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.movies[~self.movies['movie_id'].isin(items_to_ignore)].sample(n=topn, replace=True)

        return recommendations_df

