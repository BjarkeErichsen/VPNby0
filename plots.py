import numpy as np
import matplotlib.pyplot as plt
LEVEL = 1
PATHS = [f'data/rnd_{LEVEL}_600.npy', f'data/AC_{LEVEL}_600.npy', f'data/vpn_{LEVEL}_600.npy']  # random, actor criqic, VPN
# PATHS = ['data/rnd_1_600.npy', 'data/AC_1_600_wins.npy', 'data/VPN_1_600_wins.npy']  # random, actor criqic, VPN
list_of_i_episode, random = np.load(PATHS.pop(0))
A = len(list_of_i_episode)
list_of_i_episode, actor = np.load(PATHS.pop(0))
B = len(list_of_i_episode)
list_of_i_episode, vpn, _ = np.load(PATHS.pop(0))
C = len(list_of_i_episode)

min_episodes = min(A, B, C)
k = min_episodes  # plot subset of data (for different N_EPISODES) 

plt.figure(figsize=(10, 5))
plt.title(f'Comparison of models during training with random STARTS (level {LEVEL})')
plt.plot(list_of_i_episode[:k], random[:k], 'g.-', label='random')
# plt.plot(list_of_i_episode[:k], random_wins[:k], 'g.-', label='random')
plt.plot(list_of_i_episode[:k], actor[:k], 'r.-', label='actor critic')
# plt.plot(list_of_i_episode[:k], actor_wins[:k], 'r.-', label='actor critic')
plt.plot(list_of_i_episode[:k], vpn[:k], 'b.-', label='value propagation network')
# plt.plot(list_of_i_episode[:k], vpn_wins[:k], 'b.-', label='value propagation network')
plt.xlabel('Episode')
# plt.ylabel('Running average')
plt.ylabel('Winning rate')
plt.yticks([0, 0.5, 1])
plt.grid(linestyle=':')
plt.legend()
plt.show()