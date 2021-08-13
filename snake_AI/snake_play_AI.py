import torch

from snake_agent import SnakeAgent
from snake_game import SnakeGame
from snake_model import DQN


def play():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = SnakeGame(width=5, height=4, show_UI=1, gamespeed=5, player='AI')
    policy_net = DQN(game.width, game.height, 4).to(device)
    target_net = DQN(game.width, game.height, 4).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.01, total_steps=10000)
    agent = SnakeAgent(game, policy_net, target_net, optimizer, batch_size=64, replay_memory_size=10000, target_update=10, device=device)
    agent.policy_net.load('snake_model_5x4_agv14-3', agent.policy_net, device)

    previous_state = agent.get_state()
    while True:
        # Initialize the environment and state
        game.restart()
        done = 0
        while not done:
            # Get current state
            current_state = agent.get_state()
            state_memory = torch.cat([previous_state, current_state], 0)             
            previous_state = current_state 

            # Select and perform an action
            action = agent.select_action(state_memory, mode='eval')
            _, done, score = game.play_step(action.item())

            # print score
            if done:
                print(score)             
                break



if __name__ == '__main__':
    play()
