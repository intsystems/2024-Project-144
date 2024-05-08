import numpy as np
import pandas as pd
# from cmfrec import CMF_implicit
from cmfrec import CMF
import scipy.stats as sps

from helper import DistributionHandler, print_distributions
class CMFRecommender:
    def __init__(self, capacity=500, num_of_factors=40):
        self.model = CMF(k=num_of_factors)
        self.ratings = None
        self.trained = False
        self.capacity = capacity

    def get_users(self):
        return self.user_info

    def fit(self, ratings, user_info, item_info):
        self.trained = True
        self.ratings = pd.DataFrame(ratings, columns=['UserId', 'ItemId', 'Rating'])
        self.user_info = user_info
        self.item_info = item_info
        # print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        # print(self.ratings, self.user_info, self.item_info, sep='\n-------------------------------------------')
        self.model.fit(X=self.ratings, U=user_info, I=item_info)

    def get_interacted_items(self, user_id):
        return self.ratings.loc[self.ratings.UserId == user_id]['ItemId'].unique()

    def get_unqiue_item_count(self):
        return len(self.ratings["ItemId"].unique())

    def get_known_users_info(self):
        return self.user_info

    def get_known_items_info(self):
        return self.item_info

    def recommend_items_new(self, user_id, I, topn):
        recommended_items = self.model.topN_new(user=user_id, I=I, n=topn, output_score=True)
        return pd.DataFrame({"ItemId": recommended_items[0], "Rating": recommended_items[1]})

    def recommend_items_cold(self, user_row, topn=10):
        n = min(topn, self.get_unqiue_item_count())
        recommended_items = self.model.topN_cold(U=user_row, n=n, output_score=True)
        return pd.DataFrame({"ItemId": recommended_items[0], "Rating": recommended_items[1]})

    def get_max_index(self):
        return self.ratings["UserId"].max(), self.ratings["ItemId"].max()

    def recommend_items(self, user_id, items_to_recommend, topn=10, exclude_rated=True):
        items_to_ignore = []
        if exclude_rated:
            items_to_ignore.extend(self.get_interacted_items(user_id))

        items_to_recommend= np.setdiff1d(items_to_recommend, items_to_ignore)
        n = min(topn, len(items_to_recommend))
        recommended_items = self.model.topN(user=user_id, n=n, include=items_to_recommend, output_score=True)

        return pd.DataFrame({"ItemId": recommended_items[0], "Rating": recommended_items[1]})


    def retrain(self, new_ratings, new_users, new_items):
        number_of_new_ratings = len(new_ratings)

        # print("\n------------NEW RATINGS ------------------", new_ratings)
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


def zero_step(model: CMFRecommender, usefulness, z, user_info, item_info, topn=10):
    new_feedback = []

    maximal_user, maximal_item = model.get_max_index()
    old_users = model.get_known_users_info().set_index("UserId")
    old_items = model.get_known_items_info().set_index("ItemId")

    for i in range(topn):
        user_id = np.random.choice(round(maximal_user) - 1)
        for index, item_row in item_info.iterrows():
            deal = sps.bernoulli.rvs(
                usefulness(old_users.loc[user_id]["F"], item_row["F"], z))  # моделируем сделки
            new_feedback.append((user_id, item_row["ItemId"], deal))

    for index, user_row in user_info.iterrows():
        w_offered = model.recommend_items_cold(user_row["F"], topn)["ItemId"]
        for w in w_offered:
            if not (w in old_items.index.values):
                feature = item_info.loc[item_info.ItemId == w]["F"]
            else:
                feature = old_items.loc[w]["F"]
            # print((user_row["F"], feature, z))
            deal = sps.bernoulli.rvs(usefulness(user_row["F"], feature, z))  # моделируем сделки
            new_feedback.append((user_row["UserId"], w, deal))

    model.retrain(new_feedback, user_info, item_info)
    return model


def dynamic_system_iterate_CMF(model: CMFRecommender, usefulness, z, customer_distribution, w_distribution, c_size=10, w_size=10,
                           topn=8, delta=1e-4, visualize_distributions=None):
    maximal_user, maximal_item = model.get_max_index()
    user_info = pd.DataFrame(
        {"F": customer_distribution.rvs(size=c_size)})  # size = (c_size, c_feature_size) в многомерном случае
    user_info["UserId"] = np.arange(maximal_user + 1, maximal_user + 1 + c_size)

    item_info = pd.DataFrame(
        {"F": w_distribution.rvs(size=w_size)})  # size = (w_size, w_feature_size) в многомерном случае
    item_info["ItemId"] = np.arange(maximal_item + 1, maximal_item + 1 + w_size)
    model = zero_step(model, usefulness, z, user_info, item_info)

    if visualize_distributions is not None:
        print_distributions(visualize_distributions[0], visualize_distributions[1], user_info, item_info)

    new_feedback = []
    old_users = model.get_known_users_info().set_index("UserId")
    old_items = model.get_known_items_info().set_index("ItemId")

    predicted_feedback_1 = []
    predicted_feedback_2 = []

    diff_feedback = []

    for index, user_row in user_info.iterrows():
        w_offered = model.recommend_items(user_row["UserId"], item_info['ItemId'], topn=topn, exclude_rated=True)
        cur_diff_feadback = []
        predicted_cur_match_1 = []
        predicted_cur_match_2 = []

        for _, w in w_offered.iterrows():
            if not (w["ItemId"] in old_items.index.values):
                feature = item_info.loc[item_info.ItemId == w["ItemId"]]["F"]
            else:
                feature = old_items.loc[w["ItemId"]]["F"]
            u_true = usefulness(user_row["F"], feature, z)
            real_deal = sps.bernoulli.rvs(u_true)  # моделируем сделки

            new_feedback.append((user_row["UserId"], w["ItemId"], real_deal))

            predicted_deal_1 = 1 if w["Rating"] >= 0.5 else 0
            predicted_deal_2 = sps.bernoulli.rvs(w["Rating"])
            predicted_cur_match_1.append(1 if predicted_deal_1 == real_deal else 0)
            predicted_cur_match_2.append(1 if predicted_deal_2 == real_deal else 0)
            cur_diff_feadback.append(w["Rating"] - u_true)

        diff_feedback.append(np.array(cur_diff_feadback).mean())
        predicted_feedback_1.append(np.array(predicted_cur_match_1).mean())
        predicted_feedback_2.append(np.array(predicted_cur_match_2).mean())

    model.retrain(new_feedback, [], [])

    # смена распределения
    new_feedback_df = pd.DataFrame(new_feedback, columns=['UserId', 'ItemId', 'Feedback'])

    grouped_users = new_feedback_df.groupby('UserId')['Feedback'].mean().reset_index()
    grouped_users['Feedback'] += 1 / topn

    user_info = user_info.merge(grouped_users, how="inner", on='UserId')
    customer_distribution = DistributionHandler(
        sps.gaussian_kde(user_info["F"], bw_method=.1, weights=user_info['Feedback']))

    grouped_items = new_feedback_df.groupby('ItemId')['Feedback'].mean().reset_index()

    item_info = item_info.merge(grouped_items, how="left", on='ItemId').fillna(0)
    item_info['Feedback'] += delta

    w_distribution = DistributionHandler(
        sps.gaussian_kde(item_info["F"], bw_method=.1, weights=item_info['Feedback']))
    # sps.gaussian_kde(item_info["F"], weights=item_info['Feedback']))
    return customer_distribution, w_distribution, (
        np.array(predicted_feedback_1).mean(), np.array(predicted_feedback_2).mean()), sps.gaussian_kde(
        diff_feedback).pdf(0)

