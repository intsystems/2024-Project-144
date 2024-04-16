import pandas as pd
import torch
import torch.nn as nn
import scipy.stats as sps
from torch.utils.data import DataLoader
import numpy as np

from helper import DistributionHandler, print_distributions


class FeedbackDataset:
    def __init__(self, feedback: pd.DataFrame, user_info: pd.DataFrame, item_info: pd.DataFrame):
        self.feedback = feedback
        self.user_info = user_info
        self.item_info = item_info

    def __getitem__(self, idx: int):
        deal = self.feedback.iloc[idx]
        user_features = self.user_info.loc[self.user_info.UserId == deal['UserId']].drop('UserId', axis=1).to_numpy()
        item_features = self.item_info.loc[self.item_info.ItemId == deal['ItemId']].drop('ItemId', axis=1).to_numpy()
        feedback = deal["Feedback"]
        return np.hstack([user_features, item_features])[0], feedback

    def __len__(self) -> int:
        return self.feedback.shape[0]


class NeuralNetwork(nn.Module):
    def __init__(self, device, loss_function, input_shape=2, num_classes=2):
        super(self.__class__, self).__init__()
        self.ratings = []
        self.device = device
        self.loss_function = loss_function

        self.model = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Sigmoid(),
            nn.Linear(256, 32),
            nn.Sigmoid(),
            nn.Linear(32, num_classes),
            nn.Sigmoid(),
        )

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def get_interacted_items(self, user_id):
        return self.ratings.loc[self.ratings.UserId == user_id]['ItemId'].unique()

    def recommend_topN(self, user_info, items_to_recommend, topn=10):
        with torch.no_grad():
            n = min(topn, len(items_to_recommend.index))
            features = items_to_recommend["F"].to_numpy()
            x = np.vstack((np.array([user_info["F"]] * len(features)), features)).T
            x = (torch.from_numpy(x)).type(torch.FloatTensor).to(self.device)
            y_pred = self.__call__(x)[:, 1]

            res = torch.topk(y_pred, k=n)
            return pd.DataFrame({"ItemId": res.indices.cpu().numpy(), "Rating": res.values.cpu().numpy()})

    def fit_epoch(self, loader):
        self.train()
        processed_size = 0
        cumulative_loss = 0.0
        correct = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.type(torch.FloatTensor).to(self.device)
            y_batch = y_batch.type(torch.FloatTensor).to(self.device)
            self.optimizer.zero_grad()

            y_pred = self(x_batch)
            loss = self.loss_function.forward(y_pred[:, 1], y_batch)
            loss.backward()

            self.optimizer.step()
            cumulative_loss += loss
            processed_size += x_batch.shape[0]
            correct += torch.sum(torch.argmax(y_pred, dim=1) == y_batch)

        return cumulative_loss.item() / processed_size, correct.item() / processed_size

    def forward(self, inp):
        out = self.model(inp)
        return out

    def eval_epoch(self, loader):
        self.eval()
        processed_size = 0
        cumulative_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.type(torch.FloatTensor).to(self.device)
                y_batch = y_batch.type(torch.FloatTensor).to(self.device)

                y_pred = self(x_batch)
                loss = self.loss_function.forward(y_pred[:, 1], y_batch)

                cumulative_loss += loss
                processed_size += len(x_batch)
                correct += torch.sum(torch.argmax(y_pred, dim=1) == y_batch)

        return cumulative_loss.item() / processed_size, correct.item() / processed_size


def rosenblatt_test(x, y, alpha=0.05):
    rosenblatt_quantiles = {0.5: 0.12, 0.85: 0.28, 0.9: 0.35, 0.95: 0.46, 0.975: 0.58, 0.99: 0.74}
    x = np.sort(x)
    y = np.sort(y)
    all_samples = np.hstack((x, y))

    n = len(x)
    m = len(y)

    variational = sps.rankdata(all_samples, method='ordinal')

    R = variational[:n]
    S = variational[n:]

    z = (1.0 / (n + m)) * (
            1.0 / 6 + np.sum((R - np.arange(1, n + 1)) ** 2) / m + np.sum((S - np.arange(1, m + 1)) ** 2) / n) - (
                2 * n * m / (3 * (n + m)))

    expected_z = (1 + 1 / (n + m)) / 6
    variance_z = (1 + 1 / (n + m) - 3 * (n + m) / (4 * n * m)) * (1 + 1 / (n + m)) / 45

    return (z - expected_z) / np.sqrt(variance_z * 45) + 1 / 6, rosenblatt_quantiles[1 - alpha]


def dynamic_system_iterate_u(model, usefulness, z, customer_distribution, w_distribution, c_size=8, w_size=8,
                             topn=8,
                             delta=1e-4, visualize_distributions=None):
    user_info = pd.DataFrame(
        {"F": customer_distribution.rvs(size=c_size)})  # size = (c_size, c_feature_size) в многомерном случае
    user_info["UserId"] = np.arange(c_size)

    item_info = pd.DataFrame(
        {"F": w_distribution.rvs(size=w_size)})  # size = (w_size, w_feature_size) в многомерном случае
    item_info["ItemId"] = np.arange(w_size)

    if visualize_distributions is not None:
        print_distributions(visualize_distributions[0], visualize_distributions[1], user_info, item_info)

    predicted_feedback_1 = []
    predicted_feedback_2 = []
    real_feedback = []

    diff_feedback = []

    for index, user_row in user_info.iterrows():
        w_offered = model.recommend_topN(user_row, item_info, topn=topn)
        cur_diff_feadback = []
        predicted_cur_match_1 = []
        predicted_cur_match_2 = []
        for _, w in w_offered.iterrows():
            feature = item_info.loc[item_info.ItemId == w.ItemId]["F"]
            u_true = usefulness(user_row["F"], feature, z)
            real_deal = sps.bernoulli.rvs(u_true)  # моделируем сделки
            real_feedback.append((user_row["UserId"], w["ItemId"], real_deal))

            predicted_deal_1 = 1 if w["Rating"] >= 0.5 else 0
            predicted_deal_2 = sps.bernoulli.rvs(w["Rating"])
            predicted_cur_match_1.append(1 if predicted_deal_1 == real_deal else 0)
            predicted_cur_match_2.append(1 if predicted_deal_2 == real_deal else 0)
            cur_diff_feadback.append(w["Rating"] - u_true)
        diff_feedback.append(np.array(cur_diff_feadback).mean())
        predicted_feedback_1.append(np.array(predicted_cur_match_1).mean())
        predicted_feedback_2.append(np.array(predicted_cur_match_2).mean())

    new_feedback_df = pd.DataFrame(real_feedback, columns=['UserId', 'ItemId', 'Feedback'])

    batch_size = 512
    train_dataset = FeedbackDataset(new_feedback_df, user_info, item_info)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss, accuracy = model.eval_epoch(train_data_loader)
    model.fit_epoch(train_data_loader)

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
