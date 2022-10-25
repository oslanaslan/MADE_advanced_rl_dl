""" Tools """
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rc


def set_notebook_params():
    """ Set plotting parameters """
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    # rc('font',**{'family':'sans-serif'})
    # rc('text', usetex=True)
    # rc('text.latex',preamble=r'\usepackage[utf8]{inputenc}')
    # rc('text.latex',preamble=r'\usepackage[russian]{babel}')
    # rc('figure', **{'dpi': 300})


def plot_hist(data: list, title: str) -> None:
    """ Plot the histogram """
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=100)
    plt.title(title)
    plt.show()


def plot_learning_curve(data: list, title: str) -> None:
    """ Plot Q-learning curve """
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel("Games cnt")
    plt.ylabel("Reward")
    plt.yticks(np.arange(min(data), 1.1, 0.1))
    plt.plot(data, label=f"Reward")
    plt.legend()
    plt.show()
