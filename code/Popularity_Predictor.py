class PopularityRecommender:
    MODEL_NAME = 'Popularity'

    def __init__(self, ratings_train, items_df=None):
        self.popularity_df = self._calc_popularity_df(ratings_train)
        self.items_df = items_df

    def _calc_popularity_df(self, ratings_train):
        return ratings_train.groupby('movie_id')['rating'].sum().sort_values(
            ascending=False).reset_index()

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['movie_id'].isin(items_to_ignore)] \
            .sort_values('rating', ascending=False) \
            .head(topn)

        # if verbose:
        #     if self.items_df is None:
        #         raise Exception('"items_df" is required in verbose mode')
        #
        #     recommendations_df = recommendations_df.merge(self.items_df, how = 'left',
        #                                                   left_on = 'contentId',
        #                                                   right_on = 'contentId')[['eventStrength', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df

