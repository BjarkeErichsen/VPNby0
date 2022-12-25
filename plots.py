import numpy as np
import matplotlib.pyplot as plt
LEVEL = 1
PATHS = [f'data/rnd_{LEVEL}_600.npy', f'data/AC_{LEVEL}_600.npy', f'data/vpn_{LEVEL}_600.npy']  # random, actor criqic, VPN
N = 100 # number of episodes when measuring win rate (different than N_EPISODES)

# PATHS = ['data/rnd_1_600.npy', 'data/AC_1_600_wins.npy', 'data/VPN_1_600_wins.npy']  # random, actor criqic, VPN
list_of_i_episode, random = np.load(PATHS.pop(0))
A = len(list_of_i_episode)
list_of_i_episode, actor = np.load(PATHS.pop(0))
B = len(list_of_i_episode)
list_of_i_episode, vpn, _ = np.load(PATHS.pop(0))
C = len(list_of_i_episode)

min_episodes = min(A, B, C)
k = min_episodes  # plot subset of data (for different N_EPISODES) 

# https://www.statisticshowto.com/binomial-confidence-interval/#:~:text=The%20binomial%20confidence%20interval%20is,p%20for%20a%20binomial%20distribution.
# Confidence intervals - depends on win rate and number of episodes
# 
# CI = mean += 1.96 * np.sqrt(p * (1 - p) / N) IF p >= 0.1
# OTHERWISE use the Poisson approximation

# std is found by assuming win rate follows a binomial distribution, so
# var = n * p * (1 - p) = N * win_rate * (1 - win_rate) BUT NO REALLY
# std = sqrt(var)

def CI(p, N):
    return 1.96 * np.sqrt(p * (1 - p) / N)
    # std = np.sqrt(N * p * (1 - p))
    # return 1.96 * std / np.sqrt(N)




plt.figure(figsize=(10, 5))
plt.title(f'Comparison of models during training with random STARTS (level {LEVEL})')

def plot_win_rate(data, label, color):
    plt.plot(list_of_i_episode[:k], data[:k], f'{color}.-', label=label)
    ci = CI(data[:k], N)
    plt.fill_between(list_of_i_episode[:k], data[:k] - ci, data[:k] + ci, alpha=0.1, color=color)

plot_win_rate(random, 'random', 'g')
plot_win_rate(actor, 'actor critic', 'r')
plot_win_rate(vpn, 'value propagation network', 'b')

plt.xlabel('Episode')
plt.ylabel('Winning rate')
plt.yticks([0, 0.5, 1])
plt.grid(linestyle=':')
plt.legend()
plt.show()