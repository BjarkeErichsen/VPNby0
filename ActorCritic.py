# import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from GridWorld import GridWorld
from math import prod
import time
import pickle

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
N_EPISODES = 1000  # Total number of training episodes 

learning_rate = 3e-2
gamma = 0.99
seed = 0#543
delay = 0
render = True
if delay:
    render = True
log_interval = 40

wall_pct = 0.0
map = 3
map = [map]*4
non_diag = False



# env = gym.make('CartPole-v1', render_mode="rgb_array")
if seed:
    env = GridWorld(map=map, seed=seed, non_diag=non_diag, rewards=(0.0, 1.0), wall_pct=wall_pct)    
    torch.manual_seed(seed)
else:
    env = GridWorld(map=map, non_diag=non_diag, rewards=(0.0, 1.0), wall_pct=wall_pct)
# env.reset()

env.reset_to(TUHE)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        hidden_units = 32
        self.affine1 = nn.Linear(prod(env.observation_space.shape), hidden_units)

        # actor's layer
        self.action_head = nn.Linear(hidden_units, env.action_space.n)

        # critic's layer
        self.value_head = nn.Linear(hidden_units, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_logits = self.action_head(x)
        action_prob = F.softmax(action_logits, dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = state.flatten()
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns) #laver returns om til torch tensors
    # returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()  #calculating advantage, value.item() = the state value we got

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)      #

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))    #l1 smoothed absolute error

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # show updated action weights
    # print(model.get_parameter("action_head.weight"))
    # grads = []
    # for name, param in model.named_parameters():
    #     print(name, param)
        # grads.append(param.view(-1))
    # grads = torch.cat(grads)
    # print()


    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 0

    # run infinitely many episodes
    for i_episode in range(N_EPISODES): #count(1):

        # reset environment and episode reward
        state = env.reset()
        # env.render()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, n_steps_givup):

            # select action from policy
            # print(f"{i_episode}, {t} - selecting action")
            action = select_action(state)

            # take the action
            # if render:
            #     env.render()
            #     time.sleep(delay)
            state, reward, done = env.step(action)
            env.process_input()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        # print(f"{i_episode} - finishing episode")
        finish_episode()

        # log results
        if i_episode % log_interval == 0:
            print(f'Episode {i_episode} after {round((time.time() - start_time)/60, 2)} mins \
                    \tRuning reward: {round(running_reward, 2)}')
            play(1)
            if abs(running_reward - 1.00) < eps and is_solved(100):
            # if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(running_reward, t))
                break
    
def play(total_episodes):
    # env = GridWorld(map_size=(4,4,5,5), render=True, rewards=(0.0, 100.0))
    model.eval()
    env.render()
    state = env.reset()
    wins = 0
    total = 0

    # for i in range(100):
    i = 0
    while True:
        env.process_input()
        # pick best action
        state = state.flatten()
        state = torch.from_numpy(state).float()
        probs, _ = model(state)
        action = probs.argmax().item()

        # take action
        time.sleep(0.1)
        state, reward, done = env.step(action)
        env.render()

        i += 1
        if done or i > n_steps_givup:  # Complete or give up, max n_steps_givup steps
            state = env.reset()
            env.render()
            if i <= n_steps_givup: 
                wins += 1
            total += 1
            i = 0
            print(f'wins: {wins} attempts: {total}')
            if total == total_episodes:
                model.train()
                break

def is_solved(eps=100):
    """Convergence test over arg 'eps' episodes
       returns true If it can get 100 wins in a row without using 50 or more, steps
    """

    model.eval()
    state = env.reset()
    wins = 0
    total = 0

    i = 0
    while True:
        # pick best action
        state = state.flatten()
        state = torch.from_numpy(state).float()
        probs, _ = model(state)
        action = probs.argmax().item()

        # take action
        state, reward, done = env.step(action)

        i += 1
        if done:  # Complete
            state = env.reset()            
            wins += 1
            i = 0
            if wins == eps:
                model.train()
                return True

        elif i > 50:
            model.train()
            print(f'Failed evaluation: {wins}/{eps}')
            return False

if __name__ == '__main__':
    start_time = time.time()
    main()       #training the model until convergence
    pickle.dump(model, open("actor_critique_model.p", "wb"))
    play(100)        #evaluation/testing the final model, renders the output