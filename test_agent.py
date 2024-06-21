from game import Reversi
from QLearning.QLearning import QLearningAI
from Random.Random import RandomAI
from ApproximateQLearning.ApproximateQLearning import ApproximateQLearningAI, train_agent_ApproximateQLearning
from QLearning.QLearning import train_agent_QLearning
import tkinter as tk
from utils import ReversiGUI

def play_game(agent1, agent2, game_class, verbose=False):
    game = game_class()
    current_agent = agent1

    while not game.is_game_over():
        valid_moves = game.get_valid_moves(game.current_player)
        if valid_moves:
            move = current_agent.select_move(game)
            if move is not None:
                game.make_move(move[0], move[1], game.current_player)
        else:
            game.current_player = Reversi.BLACK if game.current_player == Reversi.WHITE else Reversi.WHITE

        current_agent = agent1 if game.current_player == agent1.player else agent2

    winner = game.get_winner()
    if verbose:
        if winner == agent1.player:
            print(f"({agent1.__class__.__name__}) wins!")
        elif winner == agent2.player:
            print(f"({agent2.__class__.__name__}) wins!")
        else:
            print("Draw!")

    return winner


def test_agents(agent1, agent2, game_class, games=100, gui=True):
    agent1_wins = 0
    agent2_wins = 0
    draws = 0

    if isinstance(agent1, ApproximateQLearningAI):
        print("Training ApproximateQLearningAI...")
        train_agent_ApproximateQLearning(episodes=4000, agent=agent1, opponent=RandomAI(Reversi.WHITE), game_class=Reversi)
    elif isinstance(agent1, QLearningAI):
        print("Training QLearningAI...")
        train_agent_QLearning(episodes=4000, agent=agent1, opponent=RandomAI(Reversi.WHITE), game_class=Reversi)

    if gui:
        root = tk.Tk()
        root.title("Reversi")
        gui_instance = ReversiGUI(root, agent1, agent2)

    def play_and_update():
        nonlocal agent1, agent2, agent1_wins, agent2_wins, draws, games, gui_instance

        if games > 0:
            print(f"Game {100 - games + 1}:")
            game = game_class()

            if gui:
                gui_instance.reset(game, agent1, agent2)
                root.update_idletasks()
                root.update()

                while not game.is_game_over():
                    move = agent1.select_move(game)
                    if move:
                        game.make_move(move[0], move[1], agent1.player)
                        gui_instance.update_board()
                        root.update_idletasks()
                        root.update()
                    move = agent2.select_move(game)
                    if move:
                        game.make_move(move[0], move[1], agent2.player)
                        gui_instance.update_board()
                        root.update_idletasks()
                        root.update()

                winner = game.get_winner()
                if winner == agent1.player:
                    print(f"({agent1.__class__.__name__}) wins!")
                elif winner == agent2.player:
                    print(f"({agent2.__class__.__name__}) wins!")
                else:
                    print("Draw!")
                if winner == agent1.player:
                    agent1_wins += 1
                elif winner == agent2.player:
                    agent2_wins += 1
                else:
                    draws += 1
            else:
                # 进行无 GUI 的游戏测试
                winner = play_game(agent1, agent2, game_class, verbose=True)
                if winner == agent1.player:
                    agent1_wins += 1
                elif winner == agent2.player:
                    agent2_wins += 1
                else:
                    draws += 1

            agent1, agent2 = agent2, agent1
            agent1_wins,agent2_wins = agent2_wins,agent1_wins
            games -= 1
            if gui:
                root.after(1000, play_and_update) 
            else:
                play_and_update()  
        else:
            if gui:
                root.destroy()

            print(f"Agent 1 ({agent1.__class__.__name__}) wins: {agent1_wins}")
            print(f"Agent 2 ({agent2.__class__.__name__}) wins: {agent2_wins}")
            print(f"Draws: {draws}")

    if gui:
        root.after(0, play_and_update)
        root.mainloop()
    else:
        play_and_update()

    return agent1_wins, agent2_wins, draws


if __name__ == "__main__":
    q_agent = QLearningAI(player=Reversi.BLACK)
    greedy_agent = RandomAI(player=Reversi.WHITE)

    agent1_wins, agent2_wins, draws = test_agents(q_agent, greedy_agent, Reversi, games=100, gui=False)
    print(f"Q-Learning Agent wins: {agent1_wins}")
    print(f"Random Agent wins: {agent2_wins}")
    print(f"Draws: {draws}")
