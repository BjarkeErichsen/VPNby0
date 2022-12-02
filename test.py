import numpy as np
import matplotlib.pyplot as plt


N_EPISODES = 1000
LEVEL = 0
model_names = ["AC", "VPN"]
model_indx = 0
PATH = f"{model_names[model_indx]}_{LEVEL}_{N_EPISODES}"

print(PATH)