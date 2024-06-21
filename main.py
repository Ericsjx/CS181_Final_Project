from utils import ReversiGUI
import tkinter as tk
from game import Reversi
from ApproximateQLearning.ApproximateQLearning import ApproximateQLearningAI,train_agent_ApproximateQLearning
from Greedy.Greedy import GreedyAI
from MCTS.MCTS import MCTSAgent
from Minimax.Minimax import MinimaxAI
from QLearning.QLearning import QLearningAI, train_agent_QLearning
from Random.Random import RandomAI
from test_agent import test_agents
import argparse

if __name__ == '__main__':
    # agent1 = RandomAI(Reversi.BLACK)
    # agent1 = MCTSAgent(Reversi.BLACK, simulations=1000)  
    # agent1 = ApproximateQLearningAI(Reversi.BLACK)
    # agent1 = QLearningAI(Reversi.BLACK)
    agent1 = MinimaxAI(Reversi.BLACK,depth=3)
    # agent1 = GreedyAI(Reversi.BLACK)

    agent2 = RandomAI(Reversi.WHITE) 
    # agent2 = GreedyAI(Reversi.WHITE)
    # agent2 = MinimaxAI(Reversi.WHITE,depth=3)
    #agent2 = MCTSAgent(Reversi.WHITE, simulations=1000) 
    parser = argparse.ArgumentParser(description="Reversi AI")
    parser.add_argument("--no-GUI", action="store_true", help="Run tests without GUI")
    args = parser.parse_args()
    if args.no_GUI:
        test_agents(agent1, agent2, Reversi, games=100, gui=False)
    else:
        test_agents(agent1, agent2, Reversi, games=100, gui=True)
