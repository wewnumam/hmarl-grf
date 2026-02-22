import numpy as np
import pickle
import matplotlib.pyplot as plt


def plot_learning_curve():
    rewards = np.load("results/rewards.npy")
    plt.figure()
    plt.plot(rewards, label="Episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()


def plot_qtable():
    with open("results/qtable.pkl", "rb") as f:
        q = pickle.load(f)

    xs, ys, vs = [], [], []
    for (x, y), qv in q.items():
        xs.append(x)
        ys.append(y)
        vs.append(max(qv))

    plt.figure()
    sc = plt.scatter(xs, ys, c=vs)
    plt.colorbar(sc, label="State value")
    plt.xlabel("Ball X")
    plt.ylabel("Ball Y")
    plt.title("Q-table Heatmap")
    plt.show()


if __name__ == "__main__":
    plot_learning_curve()
    plot_qtable()