import math
import random
from collections import deque, namedtuple
from itertools import count

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from IPython import display

from snake_game import SnakeGame
from snake_model import DQN

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



# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


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


class SnakeAgent():
    def __init__(self, game, lr, batch_size, target_update):
        # init game
        self.game = game

        # init nets
        self.n_actions = 4
        self.policy_net = DQN(game.width, game.height, self.n_actions).to(device)

        self.target_net = DQN(game.width, game.height, self.n_actions).to(device)    
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.target_update = target_update

        # init optimizer and memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(10000)
        
        # init training parameters
        self.batch_size = batch_size
        self.gamma = 0.9

        # exploration parameters
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.steps_done = 0



    def get_state(self):
        grid = -1*torch.ones(self.game.width, self.game.height)

        for i in range(1, len(self.game.snake)):
            grid[self.game.snake[i].x, self.game.snake[i].y] = 1

        if self.game.head.x == self.game.width:
            x_location = self.game.width - 1
        else:
            x_location = self.game.head.x

        if self.game.head.y == self.game.height:
            y_location = self.game.height - 1
        else:
            y_location = self.game.head.y

        grid[x_location, y_location] = 2
        grid[self.game.food.x, self.game.food.y] = 3

        return torch.unsqueeze(grid, 0)


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        # exploration
        if sample < eps_threshold and self.steps_done < 500:
            action = random.randint(0, 3)

        # exploitation
        else:
            state_0 = torch.unsqueeze(state, 0)
            prediction = self.policy_net(state_0)
            action = torch.argmax(prediction).item()

        return torch.tensor(action).view(1)


    def optimize_model(self):

        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.cat(batch.done)

        # Calculate Q values
        curr_Q = self.policy_net(state_batch.unsqueeze(1)).gather(1, action_batch.unsqueeze(1))
        next_Q = self.target_net(next_state_batch.unsqueeze(1)).max(1)[0].detach()
        expected_Q = reward_batch + (1 - done_batch) * (self.gamma * next_Q) 

        # Compute  loss
        criterion = nn.MSELoss()
        loss = criterion(curr_Q, expected_Q.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train():

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    num_games = 0
    record = 0

    game = SnakeGame(width=5, height=5, show_UI=0, gamespeed=100)
    agent = SnakeAgent(game, lr=0.01, batch_size=128, target_update=10)
    show_plot = False

    while True:
        # Initialize the environment and state
        game.restart()
        num_games += 1
        
        for t in count():
            # Get current state
            state = agent.get_state()
            # Select and perform an action
            action = agent.select_action(state)
            reward, done, score = game.play_step(action.item())
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([done], device=device)

            # Observe new state
            next_state = agent.get_state()

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward, done)

            # Perform one step of the optimization (on the policy network)
            agent.optimize_model()

            # if game is done store record, plot and log results
            if done:
                if score > record: 
                    record = score

                total_score += score
                mean_score = total_score / num_games

                if show_plot:
                    plot_scores.append(score)
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores)

                if num_games % agent.batch_size == 0:
                    print('Game:', num_games, '    Record in batch:', record, '    mean score:', mean_score)
                    record = 0

                break

        # Update the target network, copying all weights and biases in DQN
        if num_games % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())


if __name__ == '__main__':
    train()
