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

if __name__ == '__main__':
    #agent1 = RandomAI(Reversi.BLACK)
    #agent1 = MCTSAgent(Reversi.BLACK, simulations=1000)  # MCTS代理
    agent1 = ApproximateQLearningAI(Reversi.BLACK)
    #agent1 = MinimaxAI(Reversi.BLACK,depth=3)
    agent2 = RandomAI(Reversi.WHITE)  # Minimax代理
    #agent2=GreedyAI(Reversi.WHITE)
    #agent2 = MinimaxAI(Reversi.WHITE,depth=3)
    train_agent_ApproximateQLearning(episodes=4000,agent=agent1,opponent=agent2,game_class=Reversi)
    agent1_wins, agent2_wins, draws = test_agents(agent1, agent2, Reversi, games=100)




# if __name__ == '__main__':
#     # 启动 GUI
#     root = tk.Tk()
#     root.title("Reversi")
#     player1 = QLearningAI(Reversi.BLACK)
#     player2 = RandomAI(Reversi.WHITE)
#     gui = ReversiGUI(root, player1, player2)

#     # 开始训练并在GUI中显示
#     train_with_gui(episodes=4000, agent1=player1, agent2=player2, gui=gui)

#     root.mainloop()