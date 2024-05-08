import pandas as pd
import torch
import torch.nn as nn
import scipy.stats as sps
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from helper import DistributionHandler, print_distributions, construct_probability_density


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

    def recommend_topN(self, user_info, items_to_recommend, u_pred_case=1, topn=10):
        with torch.no_grad():
            n = min(topn, len(items_to_recommend.index))
            features = items_to_recommend["F"].to_numpy()
            x = np.vstack((np.array([user_info["F"]] * len(features)), features)).T
            x = (torch.from_numpy(x)).type(torch.FloatTensor).to(self.device)
            y_pred = self.__call__(x)[:, 1]
            if u_pred_case == 1:
                res = torch.topk(y_pred, k=n)
                return pd.DataFrame({"ItemId": res.indices.cpu().numpy(), "Rating": res.values.cpu().numpy()})
            elif u_pred_case == 2:
                res = pd.DataFrame({"ItemId": np.arange(len(features)), "Rating": y_pred.cpu().numpy()})
                res["Rating"] = res["Rating"].transform(lambda x: 1 if x >= 0.5 else 0)
                max_buy = res["Rating"].sum()
                if max_buy >= n:
                    return res[res["Rating"] == 1].sample(n=n)
                else:
                    return pd.concat([res[res["Rating"] == 0].sample(n=n - max_buy), res[res["Rating"] == 1]], axis=0)

    def print_3D(self, x, y, Z_pred, Z_true):
        X, Y = np.meshgrid(x, y)
        Z_pred = np.reshape(Z_pred, X.shape)
        Z_true = np.reshape(Z_true, X.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z_pred)
        plt.xlabel("x")
        plt.ylabel("y")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z_true)
        # plt.contour(X, Y, np.reshape(Z_true, X.shape), levels=20)
        plt.xlabel("x")
        plt.ylabel("y")

    def get_prediction_pair(self, x, y, u_pred_case=1):
        X, Y = np.meshgrid(x, y)
        positions = np.vstack([X.ravel(), Y.ravel()]).T
        Z_pred = None
        with torch.no_grad():
            positions = (torch.from_numpy(positions)).type(torch.FloatTensor).to(self.device)
            # Evaluate the probability density function at the grid points
            Z_pred = self.__call__(positions)[:, 1].cpu().numpy()
            positions = positions.cpu().numpy()

        if u_pred_case == 1:
            return positions, Z_pred
        else:
            vectorized = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
            return positions, vectorized(Z_pred)

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


def get_cartesian(x, y):
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])


def dynamic_system_iterate_u(model, usefulness, z, L, c_w_distribution, u_pred_case=1, c_size=8, w_size=8,
                             topn=8, visualize_distributions=None):
    current_sample = c_w_distribution.rvs(size=max(c_size, w_size))
    user_info = pd.DataFrame(
        {"F": current_sample[0][:c_size]})  # size = (c_size, c_feature_size) в многомерном случае
    user_info["UserId"] = np.arange(c_size)

    item_info = pd.DataFrame(
        {"F": current_sample[1][:w_size]})  # size = (w_size, w_feature_size) в многомерном случае
    item_info["ItemId"] = np.arange(w_size)

    x = user_info["F"].to_numpy()
    y = item_info["F"].to_numpy()
    points, Z_pred = model.get_prediction_pair(x, y, u_pred_case=u_pred_case)
    L_values = []
    Z_true = []
    for i, pos in enumerate(points):
        u_true = usefulness(pos[0], pos[1], z())
        Z_true.append(u_true)
        L_values.append(L(u_true, Z_pred[i]))


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
        w_offered = model.recommend_topN(user_row, item_info, u_pred_case=u_pred_case, topn=topn)
        cur_diff_feadback = []
        predicted_cur_match = []
        for _, w in w_offered.iterrows():
            feature = item_info.loc[item_info.ItemId == w.ItemId]["F"].item()

            u_true = usefulness(user_row["F"], feature, z())
            L_metric.append(L(u_true, w["Rating"]))
            # points.append((user_row["F"], feature))

            real_deal = sps.bernoulli.rvs(u_true)  # моделируем сделки
            real_feedback.append((user_row["UserId"], w["ItemId"], real_deal))
            predicted_deal = sps.bernoulli.rvs(w["Rating"])
            predicted_cur_match.append(1 if predicted_deal == real_deal else 0)

            cur_diff_feadback.append(w["Rating"] - u_true)
        diff_feedback.append(np.mean(cur_diff_feadback))
        predicted_feedback.append(np.mean(predicted_cur_match))

    new_feedback_df = pd.DataFrame(real_feedback, columns=['UserId', 'ItemId', 'Feedback'])
    batch_size = 512
    train_dataset = FeedbackDataset(new_feedback_df, user_info, item_info)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loss, _ = model.fit_epoch(train_data_loader)

    # print("\n-----------------------------------------", list(zip(points, L_values)))


    # if visualize_distributions is not None:
    #     model.print_3D(x, y, Z_pred, Z_true)

    c_w_distribution = DistributionHandler(construct_probability_density(points, np.array(L_values)))
    # debug = pd.DataFrame(points, columns=['points_x', 'points_y'])
    # debug["L"] = L_metric
    # print(debug.sort_values(by="points_y", ascending=False))
    # return c_w_distribution, np.array(predicted_feedback).mean(), loss, sps.gaussian_kde(
    #     diff_feedback).pdf(0)
    return c_w_distribution, np.array(predicted_feedback).mean(), loss, np.mean(L_metric), sps.gaussian_kde(
        diff_feedback).pdf(0)


def init_data(customer_distribution, w_distribution, start_c_size, start_w_size, usefulness, z):
    user_info = pd.DataFrame(
        {"F": customer_distribution.rvs(size=start_c_size)})  # генерим датасет для нулевой итерации
    user_info["UserId"] = np.arange(start_c_size)

    item_info = pd.DataFrame({"F": w_distribution.rvs(size=start_w_size)})
    item_info["ItemId"] = np.arange(start_w_size)
    feedback = []

    for i, user_row in user_info.iterrows():
        for j, item_row in item_info.iterrows():
            val = usefulness(user_row["F"], item_row["F"], z())
            deal = sps.bernoulli.rvs(val)
            feedback.append((user_row["UserId"], item_row["ItemId"], deal))
    feedback = pd.DataFrame(feedback, columns=['UserId', 'ItemId', 'Feedback'])
    batch_size = 512
    train_dataset = FeedbackDataset(feedback, user_info, item_info)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DistributionHandler(
        (customer_distribution, w_distribution))
