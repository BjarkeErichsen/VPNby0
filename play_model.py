import pygame as pg
import numpy as np
from GridWorld import GridWorld
import torch
from torch.distributions import Categorical
from ValuePropagationNetwork import ActorCritc, VPN

FPS = 60
GIVE_UP = 40
PATH = "AC_3_1000"
info = PATH.split("_")
LEVEL = int(info[1])
N_EPISODES = int(info[2])
pg.init()
pg.display.set_caption('GridWorld - Finished model')
pg.font.init()
clock = pg.time.Clock()


# First load the model
model = torch.load(PATH)
model.eval()

# Then create the environment
TUHE = np.array([[1, 0, 0, 0, 0, 2, 0, 1, 0, 1],
                 [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                 [1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
                 [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                 [1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                 [1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                 [0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                 [1, 0, 1, 0, 1, 3, 1, 1, 0, 1]])
env = GridWorld(map=[5]*4, non_diag=False, rewards=(0.0, 1.0), wall_pct=0.5)
s = env.reset_to(TUHE)
env.set_level(LEVEL)
env.render()

def select_action(state):
<<<<<<< HEAD
    probs, _ = model(state)

=======
    probs, _ = model(state)
    
>>>>>>> Jason
    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # the action to take (left or right)
    return action.item()

# Then run the model
wins = 0
total = 0 # keep score
step_count = 0
while True:
    # Sample an action from the model distribution
    action = select_action(s)
    
    # Step the environment
    s, r, done = env.step(action)
    step_count += 1

    # Reset if done
    if done:  # We won!
        s = env.reset()
        step_count = 0
        total += 1
        wins += 1
        pg.display.set_caption(f'GridWorld - Finished model - Win rate: {wins}/{total}')
    
    elif step_count > GIVE_UP:  # We lost
        s = env.reset()
        step_count = 0
        total += 1
        pg.display.set_caption(f'GridWorld - Finished model - Win rate: {wins}/{total}')

    # Render the environment
    env.render()
    # Process the input
    env.process_input() # <- let user quit the game
    # Wait a bit
    clock.tick(FPS)