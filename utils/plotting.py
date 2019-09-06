import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use("Agg")


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data.transpose(2, 0, 1)
    return data


def plot_far_frr_to_numpy(thresholds, far, frr):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(thresholds, far, alpha=0.5, color='green', linestyle='--', label='FAR')
    ax.plot(thresholds, frr, alpha=0.5, color='red', label='FRR')

    plt.legend(loc='upper right')

    plt.xlabel("Threshold")
    plt.xlim(0, 1)
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_roc_to_numpy(far, tpr):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(far, tpr, color='red')

    plt.xlabel("FAR")
    plt.xlim(0, 1)
    plt.ylabel("TPR")
    plt.ylim(0, 1)
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data
