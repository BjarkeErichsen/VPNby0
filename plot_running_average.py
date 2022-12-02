import numpy as np
import matplotlib.pyplot as plt

PATH = 'rndm.npy'
list_of_i_episode, list_of_running_reward = np.load(PATH)


plt.figure(figsize=(10, 5))
plt.title(f'Running average of {PATH}')
plt.plot(list_of_i_episode, list_of_running_reward, 'r.-', label='Running average')
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.grid(linestyle=':')
plt.legend()
plt.show()