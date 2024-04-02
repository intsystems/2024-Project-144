import numpy as np
import pandas as pd
from cmfrec import ContentBased

class ContentBasedRecommender:
    def __init__(self, capacity=500, num_of_factors=40):
        self.model = ContentBased(k=num_of_factors)
        self.ratings = None
        self.trained = False
        self.capacity = capacity

    def fit(self, ratings, user_info, item_info):
        self.trained = True
        self.ratings = pd.DataFrame(ratings, columns=['UserId', 'ItemId', 'Rating'])
        self.user_info = user_info
        self.item_info = item_info
        self.model.fit(X=self.ratings, U=user_info, I=item_info)

    def get_interacted_items(self, user_id):
        return self.ratings.loc[self.ratings.UserId == user_id]['ItemId'].unique()

    def get_unqiue_item_count(self):
        return len(self.ratings["ItemId"].unique())

    def get_known_users_info(self):
        return self.user_info

    def recommend_items_new(self, U, I, topn):
        recommended_items = self.model.topN_new(n=topn, U=U, I=I ,output_score=True)
        return pd.DataFrame({"ItemId": recommended_items[0], "Rating": recommended_items[1]})


    def recommend_items(self, user_id, items_to_recommend, topn=10, exclude_rated=True):
        items_to_ignore = []
        if exclude_rated:
            items_to_ignore.extend(self.get_interacted_items(user_id))

        items_to_recommend= np.setdiff1d(items_to_recommend, items_to_ignore)
        n = min(topn, len(items_to_recommend))
        recommended_items = self.model.topN(user=user_id, n=n, include=items_to_recommend, output_score=True)

        return pd.DataFrame({"ItemId": recommended_items[0], "Rating": recommended_items[1]})

    def get_max_index(self):
        return self.ratings["UserId"].max(), self.ratings["ItemId"].max()

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