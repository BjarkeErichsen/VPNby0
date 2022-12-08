from GridWorld import GridWorld
import numpy as np
import pygame

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

gamma = 0.95
def go():
    global V
    V_new = V.copy()
    for y in range(H):
        for x in range(W):
            if grid[0, y, x] or grid[2, y, x]:
                continue  # Skip walls, and goal

            # Try all actions and update using max expected reward
            for a in range(env.action_space.n):
                s1, r1 = env.visit(a, (x, y))   # Hacking the system
                Q[y, x, a] = r1 + gamma * V[s1] # Belmann's expectation equation
            assert(max(Q[y, x, :]) <= 1.0), f'Constrict within [0, 1] for color gradient'
            V_new[y, x] = max(Q[y, x, :])       # Max valued across actions
    V = V_new
    env.display_values(V)

env = GridWorld(wall_pct=0.5, seed=42, non_diag=True, space_fun=go)
# grid = env.reset()
grid = env.reset_to(TUHE)
_, H, W = grid.shape
env.render()
# V = np.random.uniform(-10, 10, (H, W))
# V = np.zeros((H, W))
V = np.load('models/VPN_1_600_wins.model_V_table.npy')
# V = np.ones(grid.shape) * 2
# V[tuple(*np.argwhere(grid[2]))] = 0.0  # Must do this for proper convergence
Q = np.empty((*V.shape, env.action_space.n))  # (action, y, x)
clock = pygame.time.Clock()
FPS = 60
env.render()
env.display_values(V)

while True:

    # Process the input
    env.process_input() # <- let user quit the game
    # Wait a bit
    clock.tick(FPS)

# while True:
#     obs = env.process_input()
#     if type(obs) == tuple:  # step return
#         grid, r, done = obs
#         env.render()
#         print(done)
#     elif type(obs) == np.ndarray:  # reset return
#         grid = obs
#         V = np.random.uniform(-10, 10, (H, W))
#         _, H, W = grid.shape
#         V = np.zeros((H, W))
#         env.render()
