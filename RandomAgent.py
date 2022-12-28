from time import sleep
from GridWorld import GridWorld
import numpy as np
# import pygame as pg
import time
import matplotlib.pyplot as plt

TUHE = np.array([[1,0,0,0,0,2,0,1,0,1],
                [1,0,0,0,0,0,1,0,1,0],
                [1,1,0,0,1,0,0,1,1,1],
                [0,0,0,0,1,0,0,1,0,1],
                [1,1,0,1,1,0,1,1,1,0],
                [1,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [1,0,1,0,0,1,0,0,0,1],
                [0,1,1,0,0,0,0,0,1,1],
                [1,0,1,0,1,3,1,1,0,1]])

GIVE_UP = 15  # Number of steps before giving up
N_EPISODES = 10_001  # Total number of training episodes 
LEVEL = 3
PATH = f"rnd_{LEVEL}_{N_EPISODES}"

# log_interval = 400

wall_pct = 0.32
map = 5
map = [map]*4
non_diag = False


env = GridWorld(map=map, non_diag=non_diag, rewards=(0.0, 1.0), wall_pct=wall_pct, max_steps=GIVE_UP)
# env.reset()

# env.reset_to(TUHE)
env.set_level(LEVEL)
# env.render()

start_time = time.time()

# running_reward = 0
# list_of_running_reward = []

total_reward = 0

# Random shit
for i_episode in range(N_EPISODES):
    state = env.reset()
    ep_reward = 0

    while True:
        # action = env.sample(True)
        action = env.action_space.sample()
        s, r, done = env.step(action)
        ep_reward += r
        # env.render()

        if done:
            total_reward += ep_reward
            break

        # env.process_input()
    
    # End of episode, log stuff
    # if i_episode % log_interval == 0:
    #     minutes = (time.time() - start_time)/60
    #     print(f'{round(i_episode / N_EPISODES, 2)}% - {round(minutes, 2)} mins \
    #         Win rate: {win_ratio}')
    #     ith_episode.append(i_episode)

# np.save(f"data/{PATH}", np.array([ith_episode, list_of_running_reward]))
print(f"Win rate: {total_reward / N_EPISODES}")  # 0.295970402959704 / 0.3000699930006999

# clock = pg.time.Clock()
# while True:
#     clock.tick(30)
#     env.render()
#     obs = env.process_input()
#     if type(obs) == tuple and obs[2]:
#         s = env.reset()
#         env.render()