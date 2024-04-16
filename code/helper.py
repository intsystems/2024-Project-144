import scipy.stats as sps
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class DistributionHandler:
    def __init__(self, distribution):
        self.distribution = distribution

    def rvs(self, size=1):
        if issubclass(type(self.distribution), sps.rv_continuous):
            return self.distribution.rvs(size)
        else:
            return self.distribution.resample(size)[0]


def print_distributions(num_of_iteration, axs, user_info, item_info):
    sns.kdeplot(user_info["F"], ax=axs[0], label=f"Itration number = {num_of_iteration}")
    axs[0].set_title("User Distribution")
    sns.kdeplot(data=item_info["F"], ax=axs[1])
    axs[1].set_title("Item Distribution")


def inverse_function(L):
    epsilon = 1e-8  # Small constant to avoid division by zero
    return 1 / (L + epsilon)


def construct_probability_density(points, L_values):
    # Calculate the inverse of L_values
    inverse_values = inverse_function(L_values)

    # Normalize the inverse values
    normalized_values = inverse_values / np.sum(inverse_values)

    # Create a Gaussian KDE object
    pts = np.vstack([point for point in points]).T
    kde = sps.gaussian_kde(pts, weights=normalized_values)
    return kde

def print_3D(p):
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Evaluate the probability density function at the grid points
    Z = np.reshape(p(positions).T, X.shape)

    plt.contourf(X, Y, Z, levels=20, cmap=cm.viridis)
    plt.colorbar(label="Probability Density")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Probability Density Function p(x, y)")
    plt.show()


def L(x, y):
    return (x - y) ** 2