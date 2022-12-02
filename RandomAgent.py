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

GIVE_UP = 40  # Number of steps before giving up
N_EPISODES = 1000  # Total number of training episodes 

log_interval = 40

wall_pct = 0.5
map = 5
map = [map]*4
non_diag = False


env = GridWorld(map=map, non_diag=non_diag, rewards=(0.0, 1.0), wall_pct=wall_pct)
# env.reset()

env.reset_to(TUHE)
# env.render()

start_time = time.time()

running_reward = 0
list_of_running_reward = []
list_of_i_episode = []
give_ups = 0

# Random shit
for i_episode in range(N_EPISODES):
    state = env.reset()
    ep_reward = 0

    for _ in range(1, GIVE_UP):
        # action = env.sample(True)
        action = env.action_space.sample()
        s, r, done = env.step(action)
        ep_reward += r
        # env.render()

        if done:
            break

        env.process_input()
    else:
        give_ups += 1
    
    running_reward = 0.02 * ep_reward + (1 - 0.02) * running_reward

    if i_episode % log_interval == 0:
            print(f'Episode {i_episode} after {round((time.time() - start_time)/60, 2)} mins \
                    \tRunning reward: {round(running_reward, 2)}')
            list_of_i_episode.append(i_episode)
            list_of_running_reward.append(running_reward)

print("Give ups: ", give_ups)
np.save("rndm", np.array([list_of_i_episode, list_of_running_reward]))

# clock = pg.time.Clock()
# while True:
#     clock.tick(30)
#     env.render()
#     obs = env.process_input()
#     if type(obs) == tuple and obs[2]:
#         s = env.reset()
#         env.render()