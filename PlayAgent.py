import pygame as pg
import numpy as np
from GridWorld import GridWorld
import torch
from torch.distributions import Categorical
from AgentTraining import ActorCritc, VPN

# PATH = "agents/AC_4_10"
MODEL = "AC"
LEVEL = 4
N_EPISODES = 10_000
PATH = f"agents/{MODEL}_{LEVEL}_{N_EPISODES}"
info = PATH.split("_")
# LEVEL = int(info[1])
# N_EPISODES = int(info[2])

FPS = 60
GIVE_UP = 15
pg.init()
# pg.display.set_caption('GridWorld - Finished model')
pg.font.init()
clock = pg.time.Clock()


# First load the model
model = torch.load(PATH)
model.eval()

# Then create the environment
# TUHE = np.array([[1, 0, 0, 0, 0, 2, 0, 1, 0, 1],
#                  [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
#                  [1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
#                  [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
#                  [1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
#                  [1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#                  [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
#                  [0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
#                  [1, 0, 1, 0, 1, 3, 1, 1, 0, 1]])
env = GridWorld(map=[5]*4, non_diag=False, rewards=(0.0, 1.0), wall_pct=0.38, max_steps=GIVE_UP)
# s = env.reset_to(TUHE)
s = env.reset()
env.set_level(LEVEL)
env.render()

def select_action(state):
    probs, _ = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # the action to take (left or right)
    return action.item()

def update_caption():
    pg.display.set_caption(f'GridWorld - FPS: {FPS} - Win rate: {wins}/{total} ~ {wins/total:.2f}')

# Then run the model
wins = 0
total = 0 # keep score

while True:
    # Sample an action from the model distribution
    action = select_action(s)
    
    # Step the environment
    s, r, done = env.step(action)

    # Reset if done
    if done:  # Game over
        s = env.reset()
        total += 1
        wins += 1 if r > 0 else 0
        update_caption()
    
    # elif step_count > GIVE_UP:  # We lost
    #     s = env.reset()
    #     step_count = 0
    #     total += 1
    #     update_caption()
        

    # Render the environment
    env.render()
    # Process the input
    env.process_input() # <- let user quit the game
    keys = pg.key.get_pressed()
    if keys[pg.K_UP]:
        FPS += 1
        update_caption()
    elif keys[pg.K_DOWN]:
        FPS -= 1
        update_caption()
    # Wait a bit
    clock.tick(FPS)