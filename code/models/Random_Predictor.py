import pandas as pd
import torch
import torch.nn as nn
import scipy.stats as sps
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d
from helper import DistributionHandler, print_distributions, construct_probability_density
from scipy.interpolate import griddata


class RandomModel():
    def __init__(self, u_pred):
        self.u_pred = u_pred
    def recommend_topN(self, item_info, topn=10):
        items = item_info.sample(n=topn)
        return pd.DataFrame({"ItemId": items["ItemId"], "Rating": self.u_pred}).reset_index(drop=True)
def dynamic_system_iterate_random(model, usefulness, z, L, c_w_distribution, u_pred_case=1, c_size=8, w_size=8,
                             topn=8, visualize_distributions=None):
    current_sample = c_w_distribution.rvs(size=max(c_size, w_size))
    user_info = pd.DataFrame(
        {"F": current_sample[0][:c_size]})  # size = (c_size, c_feature_size) в многомерном случае
    user_info["UserId"] = np.arange(c_size)

    item_info = pd.DataFrame(
        {"F": current_sample[1][:w_size]})  # size = (w_size, w_feature_size) в многомерном случае
    item_info["ItemId"] = np.arange(w_size)

    points = []
    L_values = []

    if visualize_distributions is not None:
        print_distributions(visualize_distributions[0], visualize_distributions[1], user_info, item_info,
                            current_sample)
        # model.print_3D(x, y, Z_pred, Z_true)


    predicted_feedback = []
    real_feedback = []
    L_metric = []
    # points = []
    diff_feedback = []

    for index, user_row in user_info.iterrows():
        w_offered = model.recommend_topN(item_info, topn=topn)
        # print(user_row, "\n-----------------\n", w_offered)
        cur_diff_feadback = []
        predicted_cur_match = []
        for _, w in w_offered.iterrows():
            feature = item_info.loc[item_info.ItemId == w.ItemId]["F"].item()

            u_true = usefulness(user_row["F"].item(), feature, z())
            L_metric.append(L(u_true, w["Rating"].item()))
            L_values.append(L(u_true, w["Rating"].item()))
            points.append((user_row["F"], feature))
            real_deal = sps.bernoulli.rvs(u_true)  # моделируем сделки
            real_feedback.append((user_row["UserId"].item(), w["ItemId"].item(), real_deal))
            predicted_deal = sps.bernoulli.rvs(w["Rating"].item())
            predicted_cur_match.append(1 if predicted_deal == real_deal else 0)

            cur_diff_feadback.append(w["Rating"].item() - u_true)

        for index_item, w in item_info.iterrows():
            if not w["ItemId"].item() in w_offered["ItemId"].to_numpy():
                points.append((user_row["F"].item(), w["F"].item()))
                L_values.append(1)

        diff_feedback = np.hstack([diff_feedback, cur_diff_feadback])
        predicted_feedback.append(np.mean(predicted_cur_match))

    c_w_distribution = DistributionHandler(construct_probability_density(points, np.array(L_values)))

    hst = np.histogram(diff_feedback, density=True, bins=200)
    f_t = interp1d(hst[1][:-1], hst[0], kind='linear',
                   fill_value=0.0, bounds_error=False)
    return (c_w_distribution, np.array(predicted_feedback).mean(), 0, np.mean(L_metric), f_t(0.0),
            (user_info["F"].mean(), item_info["F"].mean()), (user_info["F"].var(), item_info["F"].var()))

