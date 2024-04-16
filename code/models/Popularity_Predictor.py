import numpy as np
import pandas as pd
from cmfrec import MostPopular


class PopularityRecommender:

    def __init__(self, capacity=500):
        self.model = MostPopular()
        self.ratings = None
        self.trained = False
        self.capacity = capacity

    def _find_nearest(self, feature, mode='UserId'):
        if mode == 'UserId':
            idx = (np.abs(self.user_info['F'] - feature)).idxmin()
            return self.user_info.loc[idx, mode]
        if mode == 'ItemId':
            idx = (np.abs(self.user_info['F'] - feature)).idxmin()
            return self.user_info.loc[idx, mode]

    def recommend_topn(self, user_feature, item_features, topn=10):
        nearest_user = self._find_nearest(user_feature, mode='UserId')
        nearest_items = [self._find_nearest(item_feature, mode='ItemId') for item_feature in item_features]
        recommended = self.model.topN(nearest_user, topn=topn, include=nearest_items)
        return item_features[
            np.where(np.isin(nearest_items, recommended))]  # возвращаем те, для который ближайший был порекоммендован

    def fit(self, ratings, user_info, item_info):
        self.trained = True
        self.ratings = pd.DataFrame(ratings, columns=['UserId', 'ItemId', 'Rating'])
        self.user_info = user_info
        self.item_info = item_info
        self.model.fit(X=self.ratings, U=user_info, I=item_info)


    def retrain(self, new_ratings, new_users, new_items):
        number_of_new_ratings = len(new_ratings)
        new_ratings = pd.DataFrame(new_ratings, columns=['UserId', 'ItemId', 'Rating'])
        self.ratings = pd.concat([self.ratings, new_ratings], ignore_index=True)
        self.truncate_to_capacity()
        if len(new_users) != 0:
            self.user_info = pd.concat([self.user_info, new_users])
        if len(new_items) != 0:
            self.item_info = pd.concat([self.item_info, new_items])

        self.fit(ratings=self.ratings, user_info=self.user_info, item_info=self.item_info)

    def truncate_to_capacity(self):
        if self.ratings.shape[0] > self.capacity:
            self.ratings = self.ratings[self.ratings.shape[0] - self.capacity:]

    def get_users(self):
        return self.user_info

    def get_items(self):
        return self.item_info

    # def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
    #     # Recommend the more popular items that the user hasn't seen yet.
    #     recommendations_df = self.popularity_df[~self.popularity_df['item_id'].isin(items_to_ignore)] \
    #         .sort_values('rating', ascending=False) \
    #         .head(topn)
    #
    #     # if verbose:
    #     #     if self.items_df is None:
    #     #         raise Exception('"items_df" is required in verbose mode')
    #     #
    #     #     recommendations_df = recommendations_df.merge(self.items_df, how = 'left',
    #     #                                                   left_on = 'contentId',
    #     #                                                   right_on = 'contentId')[['eventStrength', 'contentId', 'title', 'url', 'lang']]
    #
    #     return recommendations_df
