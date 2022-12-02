from time import sleep
from GridWorld import GridWorld
import numpy as np
import pygame as pg

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

n_steps_givup = 40  # Number of steps before giving up
N_EPISODES = 100  # Total number of training episodes 

learning_rate = 3e-2
gamma = 0.99
seed = 0#543
fps = 0.5
render = False
log_interval = 40

wall_pct = 0.5
map = 5
map = [map]*4
non_diag = False



# env = gym.make('CartPole-v1', render_mode="rgb_array")
if seed:
    env = GridWorld(map=map, seed=seed, non_diag=non_diag, rewards=(0.0, 1.0), wall_pct=wall_pct)    
else:
    env = GridWorld(map=map, non_diag=non_diag, rewards=(0.0, 1.0), wall_pct=wall_pct)
# env.reset()

env.reset_to(TUHE)
env.render()
n = 0
# Random shit
# for n in range(10000000):
#     # action = env.sample(True)
#     action = env.action_space.sample()
#     env.render()
#     s, r, done = env.step(action)
#     if done:
#         env.reset()
#         env.render()
#         # n += 1
#         # print(n)
#     env.process_input()
#     sleep(0.1)

print("DONE")

clock = pg.time.Clock()
while True:
    clock.tick(30)
    env.render()
    obs = env.process_input()
    if type(obs) == tuple and obs[2]:
        s = env.reset()
        env.render()