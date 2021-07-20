import torch

from snake_agent import SnakeAgent
from snake_game import SnakeGame


def play():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = SnakeGame(width=4, height=4, show_UI=1, gamespeed=5)
    agent = SnakeAgent(game, lr=0.001, batch_size=64, replay_memory_size=10000, target_update=10, device=device)

    agent.policy_net.load('snake_model', agent.policy_net, device)


    while True:
        # Initialize the environment and state
        game.restart()
        done = 0
        while not done:
            # Get current state
            state = agent.get_state()
            # Select and perform an action
            action = agent.select_action(state, mode='eval')
            _, done, score = game.play_step(action.item())

            # if game is done store record, plot and log results
            if done:
                print(score)             
                break



if __name__ == '__main__':
    play()
