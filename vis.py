import numpy as np
from matplotlib import pyplot as plt
import pickle

step = 1000
files = [
    "greedy_pred",
    "mcts_pred",
    "ours_pred",
    "ours_pred_no_sort",
    "ours_pred_no_dedup",
]
colors = ["blue", "green", "red", "purple", "orange"]

plt.figure(figsize=(12, 9), dpi=300)
time = np.arange(step)

for file, color in zip(files, colors):
    rewards = pickle.load(open(f"{file}.pickle", "rb"))
    rewards = np.array(rewards["H"] + rewards["L"])
    mean_trajectory = np.mean(rewards, axis=0)
    std_trajectory = np.std(rewards, axis=0)

    plt.plot(time, mean_trajectory, color=color)
    plt.fill_between(time, 
                    mean_trajectory - std_trajectory, 
                    mean_trajectory + std_trajectory, 
                    color=color, alpha=0.2, label=file)

plt.xlabel("Step", fontsize=16)
plt.ylabel("Value", fontsize=16)
plt.title("LM Log Likelihood Increase", fontsize=20)
plt.legend(fontsize=12)
plt.savefig("plot.png")
plt.show()