import numpy as np
import matplotlib.pyplot as plt
#fail

def CI(p, N): return 1.96 * np.sqrt(p * (1 - p) / N)

def plot_win_rate(X, Y, test_count, label, color):
    plt.plot(X, Y, f'{color}.-', label=label)
    ci = CI(Y, test_count)
    plt.fill_between(X, Y - ci, Y + ci, alpha=0.1, color=color)

def plot_agent(data, test_count): # Used from AgentTraining after termination
    episodes, wins = data
    plt.figure(figsize=(10, 5))
    plt.title(f'Training Complete')

    plot_win_rate(episodes, wins, test_count, 'agent', 'r')

    plt.xlabel('Episode')
    plt.ylabel('Winning rate')
    plt.yticks([0, 0.5, 1])
    plt.grid(linestyle=':')
    plt.legend()
    plt.show()

def main():
    # plot_agent(np.load('data/AC_4_1200.npy'), 200)
    # return
    
    PATHS = [f'data/rnd_{LEVEL}_{N_EPISODES}.npy', f'data/AC_{LEVEL}_{N_EPISODES}.npy', f'data/vpn_{LEVEL}_{N_EPISODES}.npy']  # random, actor criqic, VPN

    # PATHS = ['data/rnd_1_600.npy', 'data/AC_1_600_wins.npy', 'data/VPN_1_600_wins.npy']  # random, actor criqic, VPN
    episodes, random = np.load(PATHS.pop(0))
    A = len(episodes)
    episodes, actor = np.load(PATHS.pop(0))
    B = len(episodes)
    episodes, vpn, _ = np.load(PATHS.pop(0))
    C = len(episodes)

    min_episodes = min(A, B, C)
    k = min_episodes  # plot subset of data (for different N_EPISODES) 

    plt.figure(figsize=(10, 5))
    plt.title(f'Comparison of models during training with random STARTS (level {LEVEL})')

    plot_win_rate(episodes[:k], random[:k], N, 'random', 'g')
    plot_win_rate(episodes[:k], actor[:k], N, 'actor critic', 'r')
    plot_win_rate(episodes[:k], vpn[:k], N, 'value propagation network', 'b')

    plt.xlabel('Episode')
    plt.ylabel('Winning rate')
    plt.yticks([0, 0.5, 1])
    plt.grid(linestyle=':')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    N = 200 # number of episodes when measuring win rate (different than N_EPISODES)
    LEVEL = 1
    N_EPISODES = 600
    main()