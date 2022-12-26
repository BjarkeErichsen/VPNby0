import numpy as np
from itertools import count
from collections import namedtuple
# import matplotlib.pyplot as plt
from Plots import plot_agent


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
torch.autograd.set_detect_anomaly(True)
from GridWorld import GridWorld
from math import prod
import time

# Map = np.array([[1, 0, 0, 0, 0, 2, 0, 1, 0, 1],
#                  [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
#                  [1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
#                  [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
#                  [1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
#                  [1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#                  [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
#                  [0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
#                  [1, 0, 1, 0, 1, 3, 1, 1, 0, 1]])

# Map = np.array([
#     [1, 0, 0, 0, 2],
#     [1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1],
#     [0, 1, 1, 0, 0],
#     [1, 0, 1, 3, 1]
#     ])

GIVE_UP = 15  # Number of steps before giving up  #max steps allowed in train2
#n_step is also the number of states saved to the memory buffer before deletion
N_EPISODES = 10_000 + 1  # Total number of training episodes
LEVEL = 4
MAP_SIZE = 5
TEST_COUNT = 200  # Number of test episodes
log_interval = 400
do_intermediate_tests = True

K = 5#10 #num planning iterations
learning_rate = 0.001
gamma = 0.99
seed = 0  # 543
max_allowed_steps = GIVE_UP #max steps allowed in test
regu_scaler = 0.002
fps = 0

loss_coefficients = {"value":1, "policy":1}

wall_pct = 0.32
map = MAP_SIZE  # map_size (square)
map = [map] * 4
non_diag = False
render = False

if seed:
    env = GridWorld(map=map, seed=seed, non_diag=non_diag, rewards=(0.0, 1.0),
                    wall_pct=wall_pct, max_steps=GIVE_UP)
    torch.manual_seed(seed)
else:
    env = GridWorld(map=map, non_diag=non_diag, rewards=(0.0, 1.0), 
                    wall_pct=wall_pct, max_steps=GIVE_UP)

# env.reset()
env.set_level(LEVEL)

# env.reset_to(Map)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class VPN(nn.Module):
    """Maps input to latent space?"""
    def __init__(self):
        super(VPN, self).__init__()
        hidden_units = 32
        hidden_units2 = 64
        hidden_units_policy1 = 32

        self.n_observation1 = env.observation_space.shape[1]
        self.n_observation2 = env.observation_space.shape[2]
        n_state_dims = self.n_observation1*self.n_observation2
        n_actions = len(env.DIRS)

        #input should contain
        self.affine1 = nn.Linear(prod(env.observation_space.shape), hidden_units)
        self.affine2 = nn.Linear(hidden_units, hidden_units2)

        # r_outs's head
        self.r_out = nn.Linear(hidden_units2, n_state_dims)

        # r_in's head
        self.r_in = nn.Linear(hidden_units2, n_state_dims)

        # transition probability head
        self.p = nn.Linear(hidden_units2, n_state_dims)

        #policy network stuff
        self.policyNetwork1 = nn.Linear(n_state_dims*4, hidden_units_policy1) #3 because we dont use the transition probabilities
        self.policyHead = nn.Linear(hidden_units_policy1, n_actions)


        # action & reward buffer
        self.saved_actions = []
        self.saved_probabilities_of_actions = []
        self.rewards = []
        self.shape_of_board = (env.observation_space.shape[1], env.observation_space.shape[2])
        self.v_current = torch.zeros(self.shape_of_board)
        self.v_next = torch.zeros(self.shape_of_board)

        #self.values = np.zeros(())
    def forward(self, x):
        """Assumes x to be a (3, i, j) shape"""
        current_position = (x[1]==1).nonzero()
        x = x.flatten()

        x = torch.from_numpy(x).float()
        state = x
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        r_out = torch.sigmoid(self.r_out(x))

        r_out = torch.reshape(r_out, self.shape_of_board)


        r_in = torch.sigmoid(self.r_in(x))
        r_in = torch.reshape(r_in, self.shape_of_board)


        p = torch.sigmoid(self.p(x))
        p = torch.reshape(p, self.shape_of_board)

        #value iteration

        #For all neigborhoods for all states, we define the value of the state, as the value of having taking the best action
        #We do this for K times
        #Notably, because we do this for all states we can get information from states infinitely long away!

        self.v = torch.zeros(self.shape_of_board)
        
        # Padding all grids with zeros
        v= F.pad(torch.zeros(self.shape_of_board), (1,1,1,1))
        p     = F.pad(p, (1,1,1,1))
        r_in  = F.pad(r_in, (1,1,1,1))
        r_out = F.pad(r_out, (1,1,1,1))

        for k in range(K):
            i = 0
            helper = torch.zeros((8, self.n_observation1, self.n_observation2)) # 8 directions
            # helper = torch.zeros((9, self.n_observation1, self.n_observation2)) # Stay + 8 directions
            for i_dot, j_dot in env.DIRS:  # For all directions (env uses 0 dim as x and 1 dim as y)

                #logic of indexing: Applied the same for v, p, r_in, r_out
                #we take the padded x, index only the "inner" v by 1:1+shape_of_board, then
                #move the "square" we index in the direction of i_dot, j_dot
                xs, xe = j_dot+1, 1+j_dot+self.shape_of_board[0]  # +1 because of padding
                ys, ye = i_dot+1, 1+i_dot+self.shape_of_board[1]
                # helper[i] = v[j_dot+1:1+j_dot+self.shape_of_board[0],     i_dot+1:1+i_dot+self.shape_of_board[1]] *  \
                #             p[j_dot+1:1+j_dot+self.shape_of_board[0],     i_dot+1:1+i_dot+self.shape_of_board[1]] +  \
                #             r_in[j_dot+1:1+j_dot+self.shape_of_board[0],  i_dot+1:1+i_dot+self.shape_of_board[1]] - \
                #             r_out[j_dot+1:1+j_dot+self.shape_of_board[0], i_dot+1:1+i_dot+self.shape_of_board[1]]
                helper[i] = v[xs:xe,     ys:ye] *  \
                            p[xs:xe,     ys:ye] +  \
                            r_in[xs:xe,  ys:ye] - \
                            r_out[xs:xe, ys:ye]
                i +=1
            # just the previous v without the padding
            # helper[8] = v[1:1+self.shape_of_board[0],   1:1+self.shape_of_board[1]]  # Standing still - neccesary?

            v = helper.max(dim=0)[0]  # max over the neighborhood
            if k < K-1:  # don't pad if its the last round
                v = F.pad(v, (1,1,1,1))

        #policy
        input_to_policy = torch.cat((v.flatten(), state), 0)
        action_logits = F.relu(self.policyNetwork1(input_to_policy))
        action_logits = self.policyHead(action_logits)
        action_prob = F.softmax(action_logits, dim=-1)

        #value at current state

        state_value = v[current_position]

        return action_prob, state_value # comment when extracting V
        return action_prob, state_value, v 


class ActorCritc(nn.Module):
    """implements both actor and critic in one model"""

    def __init__(self):
        super(ActorCritc, self).__init__()
        hidden_units = 32
        hidden_units2 = 64
        self.affine1 = nn.Linear(prod(env.observation_space.shape), hidden_units)
        self.affine2 = nn.Linear(hidden_units, hidden_units2)

        # actor's layer
        self.action_head = nn.Linear(hidden_units2, env.action_space.n)

        # critic's layer
        self.value_head = nn.Linear(hidden_units2, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.saved_probabilities_of_actions = []

    def forward(self, x):
        """forward of both actor and critic"""
        x = x.flatten()
        x = torch.from_numpy(x).float()

        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

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


def select_action(state):
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    model.saved_probabilities_of_actions.append(probs)

    # the action to take (left or right)
    return action.item()

def finish_episode(i=0):
    """Training code. Calculates actor and critic loss and performs backprop."""

    R = 0
    saved_actions = model.saved_actions
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values
    saved_probs = model.saved_probabilities_of_actions
    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)  # laver returns om til torch tensors
    # returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), saved_probs, R in zip(saved_actions,saved_probs, returns):
        advantage = R - value.item()  # calculating advantage, value.item() = the state value we got

        # calculate actor (policy) loss
        entropy_regularization = torch.sum(torch.log2(saved_probs)*saved_probs)  #regularization

        policy_losses.append(-log_prob * advantage + regu_scaler*entropy_regularization)  #policy loss

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))  # l1 smoothed absolute error

        """
        if i % 1000 == 0:
            print("entropy regu",  regu_scaler*entropy_regularization, "policy", -log_prob * advantage, "value", F.smooth_l1_loss(value, torch.tensor([R])))
            i+=1"""
    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum()*loss_coefficients["policy"] + torch.stack(value_losses).sum()*loss_coefficients["value"]
    # print(torch.stack(policy_losses).sum()*loss_coefficients["policy"], torch.stack(value_losses).sum()*loss_coefficients["value"])

    # perform backprop

    loss.backward(retain_graph=True)   #added retain_graph=True because of regularization
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
    del model.saved_probabilities_of_actions[:]

def main():
    test_wins = []
    ith_episode = []
    episode_rewards = []

    # Running average
    episode_n_we_average = 50
    running_reward = 0
    list_of_running_reward = []
    list_of_i_episode = []

    running_reward_random = 0
    episode_rewards_RANDOM = []
    list_of_running_reward_RANDOM = []



    for i_episode in range(N_EPISODES):  # count(1):
        # Run an entire episode for agent and another for random agent

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning

        # for t in range(1, GIVE_UP):  # REMOVE HERE: give up is placed in the environment
        while True:
            action = select_action(state)
            state, reward, done = env.step(action)

            model.rewards.append(reward)
            ep_reward += reward

            if done:
                break

        episode_rewards.append(ep_reward)

        # if i_episode >= episode_n_we_average:
        #     running_reward = sum(episode_rewards[-episode_n_we_average:-1]) / \
        #                      len(episode_rewards[-episode_n_we_average:-1])
        #     list_of_running_reward.append(running_reward)

        # perform backprop
        finish_episode()

        # Random agent
        # state = env.reset()
        # ep_reward = 0
        # for _ in range(1, GIVE_UP):
        #     action = env.action_space.sample()
        #     _, r, done = env.step(action)
        #     ep_reward += r
        #     if done:
        #         break

        # episode_rewards_RANDOM.append(ep_reward)
        # if i_episode >= episode_n_we_average:
        #     running_reward_random = sum(episode_rewards_RANDOM[-episode_n_we_average:-1]) / len(episode_rewards_RANDOM[-episode_n_we_average:-1])
        #     list_of_running_reward_RANDOM.append(running_reward_random)
        #     list_of_i_episode.append(i_episode)


        # Testing for wins
        if do_intermediate_tests:
            if i_episode % log_interval == 0 and i_episode > 0:
                win_ratio = test(TEST_COUNT) / TEST_COUNT
                # win_ratio = test_render(TEST_COUNT) / TEST_COUNT
                test_wins.append(win_ratio)
                ith_episode.append(i_episode)
                minutes = (time.time() - start_time)/60
                time_left = round((minutes / i_episode) * (N_EPISODES - i_episode), 2)
                # time_left = round(minutes * (N_EPISODES / i_episode), 2)

                print(f'{i_episode / N_EPISODES}% - {round(minutes, 2)} mins ({time_left} left) - Win rate: {win_ratio}')
        if i_episode % 10 == 0:
            print(f'Episode {i_episode} - {(time.time() - start_time)/60} mins')
            # if running_reward > 0.5:  # RAiSING THE LEVEL HERE
            #     env.level_up()
            #     print("LEVEL UP")

            # Test the agent for data collection

            #     break
    print(f'Done after {round((time.time() - start_time)/60, 2)} mins')
    results = np.array([ith_episode, test_wins])
    plot_agent(results, TEST_COUNT)
    np.save(f'data/{PATH}', results)

def test(trials=10):
    """Return win over trials"""

    model.eval()
    state = env.reset()
    wins = 0
    total = 0

    while True:
        # pick best action and take action
        action = select_action(state)
        state, r, done = env.step(action)

        if done:  # Game over
            wins += 1 if r > 0 else 0
            total += 1
            
            if total == trials:
                model.train()
                return wins
            state = env.reset()

def test_render(trials=10):
    """Return win over trials and rendering"""

    model.eval()
    state = env.reset()
    env.render()
    wins = 0
    total = 0

    while True:
        # pick best action
        action = select_action(state)

        # take action
        state, r, done = env.step(action)
        env.render()

        if done:  # Game over
            wins += 1 if r > 0 else 0
            total += 1
            
            if total == trials:
                model.train()
                env.close_display()
                return wins
            state = env.reset()
            env.render()
    
models = [ActorCritc, VPN]
model_names = ["AC", "VPN"]
MODEL_INDEX = 1

if __name__ == '__main__':

    PATH = f"{model_names[MODEL_INDEX]}_{LEVEL}_{N_EPISODES}"
    start_time = time.time()
    model = models[MODEL_INDEX]() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    eps = np.finfo(np.float32).eps.item()

    main()  # training the model until convergence
    torch.save(model, f"agents/{PATH}")
    #play()  # evaluation/testing the final model, renders the output

