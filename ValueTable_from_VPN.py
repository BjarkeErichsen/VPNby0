import pygame as pg
import numpy as np
from GridWorld import GridWorld
import torch
from torch.distributions import Categorical
from AgentTraining import ActorCritc, VPN
print("modules loaded")

FPS = 60
PATH = "models/VPN_1_600_wins.model"

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
env.reach = None
env.render()
print("Ready")
probs, _, V = model(s)
V = V.detach().numpy()
np.save(f'{PATH}_V_table', V)
env.render()
env.display_values(V)

while True:

    # Process the input
    env.process_input() # <- let user quit the game
    # Wait a bit
    clock.tick(FPS)