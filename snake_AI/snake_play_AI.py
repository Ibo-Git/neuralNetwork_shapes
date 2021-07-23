import torch

from snake_agent import SnakeAgent
from snake_game import SnakeGame


def play():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = SnakeGame(width=4, height=4, show_UI=1, gamespeed=5)
    agent = SnakeAgent(game, lr=0.01, batch_size=64, replay_memory_size=10000, target_update=10, device=device)
    #game = SnakeGame(width=10, height=10, show_UI=1, gamespeed=5)
    #agent = SnakeAgent(game, lr=0.00035, batch_size=128, replay_memory_size=10000, target_update=10, device=device)
    agent.policy_net.load('snake_model_4x4_almost_perfect', agent.policy_net, device)

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
