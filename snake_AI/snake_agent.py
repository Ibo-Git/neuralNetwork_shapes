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
    def __init__(self, game, policy_net, target_net, optimizer, batch_size, replay_memory_size, target_update, device):
        # init game
        self.game = game
        self.device = device

        # init nets
        self.policy_net = policy_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.target_update = target_update

        # init optimizer and memory
        self.optimizer = optimizer
        self.replay_memory_size = replay_memory_size
        self.memory = ReplayMemory(replay_memory_size)
        
        # init training parameters
        self.batch_size = batch_size
        self.gamma = 0.9

        # exploration parameters
        self.eps_start = 0.9
        self.eps_end = 0
        self.eps_decay = 500
        self.steps_done = 0



    def get_state(self):

        grid_val = -10
        food_val = 1000
        snake_val = 1
        head_val = 10

        grid = grid_val*torch.ones(self.game.width, self.game.height)

        for i in range(1, len(self.game.snake)):
            grid[self.game.snake[i].x, self.game.snake[i].y] = snake_val

        if self.game.head.x == self.game.width:
            x_location = self.game.width - 1
        else:
            x_location = self.game.head.x

        if self.game.head.y == self.game.height:
            y_location = self.game.height - 1
        else:
            y_location = self.game.head.y

        grid[x_location, y_location] = head_val
        grid[self.game.food.x, self.game.food.y] = food_val

        return torch.unsqueeze(grid, 0)


    def select_action(self, state, mode='train'):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        # exploration
        if sample < eps_threshold and self.steps_done < self.replay_memory_size and mode == 'train':
            action = random.randint(0, 3)

        # exploitation
        else:
            state_0 = torch.unsqueeze(state, 0)
            prediction = self.policy_net(state_0.to(self.device))
            action = torch.argmax(prediction).item()

        return torch.tensor(action).view(1)


    def optimize_model(self):

        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)

        # Calculate Q values
        curr_Q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_Q = self.target_net(next_state_batch).max(1)[0].detach()
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
    total_score_batch = 0
    total_steps_per_game = 0
    record = 0

    n_actions = 4
    lr = 0.05
    num_total_games = 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = SnakeGame(width=5, height=4, show_UI=0, gamespeed=1000)
    policy_net = DQN(game.width, game.height, n_actions).to(device)
    target_net = DQN(game.width, game.height, n_actions).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=num_total_games)
    agent = SnakeAgent(game, policy_net, target_net, optimizer, batch_size=64, replay_memory_size=10000, target_update=10, device=device)
    show_plot = False

    for num_games in range(num_total_games):
        # Initialize the environment and state
        game.restart()
        
        previous_state = agent.get_state()
        for t in count():
            # Get current state
            current_state = agent.get_state()
            state_memory = torch.cat([previous_state, current_state], 0)             

            # Select and perform an action
            action = agent.select_action(state_memory)
            reward, done, score = game.play_step(action.item())
            reward = torch.tensor([reward], device=device)
            done = torch.tensor([done], device=device)

            # Observe new state
            next_state = agent.get_state()
            next_state_memory = torch.cat([current_state, next_state], 0)             

            # update previous state
            previous_state = current_state

            # Store the transition in memory
            agent.memory.push(state_memory, action, next_state_memory, reward, done)

            # Perform one step of the optimization (on the policy network)
            agent.optimize_model()

            # if game is done
            if done:
                # adjust learning rate
                scheduler.step()

                # store record and save net
                if score > record: 
                    record = score

                # plot results                                    
                if show_plot:
                    total_score += score
                    mean_score = total_score / num_games
                    plot_scores.append(score)
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores)

                # log results
                total_steps_per_game += t
                total_score_batch += score
                if num_games % agent.batch_size == 0:
                    mean_score = total_score_batch / agent.batch_size
                    mean_steps_per_game = total_steps_per_game / agent.batch_size

                    print(
                        'Game:', num_games, 
                        '    Record in batch:', record, 
                        '    mean score batch:', mean_score, 
                        '    mean steps per game:', mean_steps_per_game, 
                        '    Learning rate:', optimizer.param_groups[0]['lr']
                    )
                    
                    record = 0
                    total_score_batch = 0
                    total_steps_per_game = 0
                    agent.policy_net.save(file_name = 'snake_model_5x5_2.pth')
                
                break

        # Update the target network, copying all weights and biases in DQN
        if num_games % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())


if __name__ == '__main__':
    train()
