import math
import random
from collections import deque, namedtuple
from itertools import count

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import flatten
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

from snake_game import SnakeGame

game = SnakeGame(width=6, height=6, show_UI=0, gamespeed=10000)
BATCH_SIZE = 256
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
TARGET_UPDATE = 1


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()

        self.kernel_size_1 = 2
        self.kernel_size_2 = 2
        self.stride_1 = 2
        self.stride_2 = 1
        self.padding_1 = 0
        self.padding_2 = 0

        self.channel_output_conv_1 = 16
        self.channel_output_conv_2 = 32

        convw = self.conv2d_size_out( self.conv2d_size_out( w, self.kernel_size_1, self.stride_1, self.padding_1 ), self.kernel_size_2, self.stride_2, self.padding_2 )
        convh = self.conv2d_size_out( self.conv2d_size_out( h, self.kernel_size_1, self.stride_1, self.padding_1 ), self.kernel_size_2, self.stride_2, self.padding_2 )
        linear_input_size = self.channel_output_conv_2 * convw * convh

        self.net = nn.Sequential(
            nn.Conv2d(1, self.channel_output_conv_1, kernel_size=self.kernel_size_1, stride=self.stride_1, padding=self.padding_1),
            nn.BatchNorm2d(self.channel_output_conv_1),
            nn.ReLU(),
            nn.Conv2d(self.channel_output_conv_1, self.channel_output_conv_2, kernel_size=self.kernel_size_2, stride=self.stride_2, padding=self.padding_2),
            nn.BatchNorm2d(self.channel_output_conv_2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_input_size, 256),
            nn.Linear(256, outputs)
        )
  
    def forward(self, x):
        x = self.net(x)
        return x
    
    def conv2d_size_out(self, size, kernel_size, stride, padding):
            return ((size - kernel_size + 2*padding) // stride ) + 1

def get_screen(game):
    grid = torch.zeros(game.width, game.height)

    for i in range(1, len(game.snake)):
        grid[game.snake[i].x, game.snake[i].y] = 1

    if game.head.x == game.width:
        x_location = game.width - 1
    else:
        x_location = game.head.x

    if game.head.y == game.height:
        y_location = game.height - 1
    else:
        y_location = game.head.y
    #if game.head.x = 0:
    #    game.head.x = 0
    #if game.head.y < 0:
    #    game.head.y = 0

    grid[x_location, y_location] = 2
    grid[game.food.x, game.food.y] = 3

    return torch.unsqueeze(grid, 0)





# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen(game)
_, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = 4

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold and steps_done < 300:
        action = random.randint(0, 3)
    # exploitation
    else:
        state_0 = torch.unsqueeze(state, 0)
        prediction = policy_net(state_0)
        action = torch.argmax(prediction).item()

    return torch.tensor(action).view(1)




episode_durations = []



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    # compute Q values
    state_action_values = policy_net(state_batch.unsqueeze(1)).gather(1, action_batch.unsqueeze(1)) # Q current state

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states.unsqueeze(1)).max(1)[0].detach()    # Q next state

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # target Q next state

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    #for param in policy_net.parameters():
    #    param.grad.data.clamp_(-1, 1)
    optimizer.step()


import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    #display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


plot_scores = []
plot_mean_scores = []
total_score = 0
num_games = 0
record = 0

while True:
    # Initialize the environment and state
    game.restart()
    num_games += 1
    
    for t in count():
        state = get_screen(game)
        # Select and perform an action
        action = select_action(state)
        reward, done, score = game.play_step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = get_screen(game)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            if score > record: record = score
            if num_games % 100 == 0:
                print('Game: ', num_games, 'Score: ', score, 'Record: ', record)
            
            #plot_scores.append(score)
            #total_score += score
            #mean_score = total_score / num_games
            #plot_mean_scores.append(mean_score)
            #plot(plot_scores, plot_mean_scores)
            #episode_durations.append(num_games + 1)
            break

    # Update the target network, copying all weights and biases in DQN
    if num_games % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

