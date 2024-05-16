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


class OracleModel():

    def recommend_topN(self, u_true_values, topn=10):
        ind = np.argsort(u_true_values)[::-1][:topn]
        return pd.DataFrame({"ItemId": ind, "Rating": u_true_values[ind]})


    def fit_epoch(self, loader):
        return [0], 0


    def eval_epoch(self, loader):
        return [0], 0

def dynamic_system_iterate_oracle(model, usefulness, z, L, c_w_distribution, u_pred_case=1, c_size=8, w_size=8,
                             topn=8, visualize_distributions=None):
    current_sample = c_w_distribution.rvs(size=max(c_size, w_size))
    user_info = pd.DataFrame(
        {"F": current_sample[0][:c_size]})  # size = (c_size, c_feature_size) в многомерном случае
    user_info["UserId"] = np.arange(c_size)

    item_info = pd.DataFrame(
        {"F": current_sample[1][:w_size]})  # size = (w_size, w_feature_size) в многомерном случае
    item_info["ItemId"] = np.arange(w_size)

    points = []

    if visualize_distributions is not None:
        print_distributions(visualize_distributions[0], visualize_distributions[1], user_info, item_info,
                            current_sample)

    predicted_feedback = []
    real_feedback = []
    L_metric = []
    diff_feedback = []

    for index_i, user_row in user_info.iterrows():
        u_true_values = usefulness(user_row["F"], item_info["F"].to_numpy(), sps.norm(0, 0.5).rvs(item_info.shape[0]))
        w_offered = model.recommend_topN(u_true_values, topn=topn)
        cur_diff_feadback = []
        predicted_cur_match = []
        for index_j, w in w_offered.iterrows():
            feature = item_info.loc[item_info.ItemId == w.ItemId]["F"].item()
            u_true = w["Rating"].item()
            L_metric.append(0)
            points.append((user_row["F"], feature))
            # print(user_row["F"].item(), u_true,  w["Rating"].item())
            real_deal = sps.bernoulli.rvs(u_true)  # моделируем сделки
            real_feedback.append((user_row["UserId"].item(), w["ItemId"].item(), real_deal))
            predicted_deal = sps.bernoulli.rvs(w["Rating"].item())
            predicted_cur_match.append(1 if predicted_deal == real_deal else 0)

            cur_diff_feadback.append(w["Rating"].item() - u_true)
        diff_feedback = np.hstack([diff_feedback, cur_diff_feadback])
        predicted_feedback.append(np.mean(predicted_cur_match))



    c_w_distribution = DistributionHandler(construct_probability_density(points, np.array(L_metric)))


    hst = np.histogram(diff_feedback, density=True)
    f_t = interp1d(hst[1][:-1], hst[0], kind='linear',
                   fill_value=0.0, bounds_error=False)
    return (c_w_distribution, np.array(predicted_feedback).mean(), 0, np.mean(L_metric), f_t(0.0),
            (user_info["F"].mean(), item_info["F"].mean()), (user_info["F"].var(), item_info["F"].var()))


