import numpy as np
import pandas as pd
from cmfrec import CMF_implicit



class CMFRecommender:
    def __init__(self, num_of_factors=40):
        self.model = CMF_implicit(k=num_of_factors)
        self.ratings = None
        self.trained = False

    def fit(self, ratings, user_info, item_info):
        self.trained = True
        self.ratings = ratings
        self.user_info = user_info
        self.item_info = item_info
        self.model.fit(X=ratings, U=user_info, I=item_info)

    def get_interacted_items(self, user_id):
        return self.ratings.loc[self.ratings.UserId == user_id]['ItemId'].unique()

    def get_unqiue_item_count(self):
        return len(self.ratings["ItemId"].unique())

    def recommend_items_new(self, user_id, I, topn):
        recommended_items = self.model.topN_new(user=user_id, n=topn, output_score=True)[:2]
        return pd.DataFrame({"ItemId": recommended_items[0], "Rating": recommended_items[1]})

    def recommend_items_cold(self, user_row, topn=10):
        n = min(topn, self.get_unqiue_item_count())
        recommended_items = self.model.topN_cold(U=user_row, n=n, output_score=True)[:2]
        return pd.DataFrame({"ItemId": recommended_items[0], "Rating": recommended_items[1]})

    def get_max_index(self):
        return self.ratings["UserId"].max(), self.ratings["ItemId"].max()

    def recommend_items(self, user_id=None, topn=10, exclude_rated=True):
        items_to_ignore = []
        if exclude_rated:
            items_to_ignore.extend(self.get_interacted_items(user_id))
        n = min(topn, self.get_unqiue_item_count() - len(items_to_ignore))

        recommended_items = self.model.topN(user=user_id, n=n, exclude=items_to_ignore, output_score=True)[:2]

        return pd.DataFrame({"ItemId": recommended_items[0], "Rating": recommended_items[1]})

    def retrain(self, new_ratings, new_users, new_items):
        number_of_new_ratings = len(new_ratings)
        new_ratings = pd.DataFrame(new_ratings, columns =['UserId', 'ItemId', 'Rating'])
        self.ratings = pd.concat([self.ratings.loc[number_of_new_ratings:], new_ratings], ignore_index=True)
        self.user_info = pd.concat([self.user_info, new_users])
        self.item_info = pd.concat([self.item_info, new_items])

        self.fit(ratings=self.ratings, user_info=self.user_info, item_info=self.item_info)