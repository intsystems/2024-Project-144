import pandas as pd
import scipy.stats as sps
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections.abc import Iterable


class DistributionHandler:
    def __init__(self, distribution):
        self.distribution = distribution

    def rvs(self, size=1):
        if issubclass(type(self.distribution), sps.rv_continuous):
            return self.distribution.rvs(size)
        elif isinstance(self.distribution, Iterable):
            values = []
            for value in self.distribution:
                values.append(value.rvs(size))
            return np.array(values)
        else:
            return self.distribution.resample(size)


def print_distributions(num_of_iteration, axs, user_info, item_info, c_w_sample):
    sns.kdeplot(user_info["F"], ax=axs[0], label=f"Iteration number = {num_of_iteration}")
    axs[0].set_title("User Distribution")
    sns.kdeplot(data=item_info["F"], ax=axs[1])
    axs[1].set_title("Item Distribution")
    # axs[1].set_ylim(0, 10)
    plt.figure()
    sns.kdeplot(data=pd.DataFrame({"CustomerF": c_w_sample[0], "ItemF": c_w_sample[1]}), x="CustomerF", y="ItemF", cmap=cm.viridis)


def inverse_function(L):
    epsilon = 1e-8  # Small constant to avoid division by zero
    return 1/(L + 0.001)


def construct_probability_density(points, L_values):
    # Calculate the inverse of L_values
    inverse_values = inverse_function(L_values)

    # Normalize the inverse values
    normalized_values = inverse_values / np.sum(inverse_values)

    # Create a Gaussian KDE object
    pts = np.vstack([point for point in points]).T
    # kde = sps.gaussian_kde(pts, bw_method=0.2, weights=normalized_values)
    kde = sps.gaussian_kde(pts, bw_method=0.1, weights=normalized_values)

    return kde

# class LValuesHandler():
#     def __init__(self):
#         self.L_values = np.array([])
#         self.points = []
#     def append(self, points, L_values, alpha=1):
#         self.points = self.points + points
#         memory = 3
#         self.L_values = np.concatenate((self.L_values / alpha, L_values))
#         if len(self.L_values) > len(L_values) * memory:
#             self.points = self.points [-len(L_values) * memory:]
#             self.L_values = np.array(self.L_values[-len(L_values) * memory:])
#         return self.points, self.L_values





def L(x, y):
    return (x - y) ** 2