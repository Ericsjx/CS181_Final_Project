from game import Reversi
from Greedy.Greedy import GreedyAI
from MCTS.MCTS import MCTSAgent
from Minimax.Minimax import MinimaxAI
from Random.Random import RandomAI
from test_agent import test_agents

if __name__ == '__main__':
    agent1 = RandomAI(Reversi.BLACK)
    #agent1 = MCTSAgent(Reversi.BLACK, simulations=1000)  # MCTS代理
    #agent1 = MinimaxAI(Reversi.BLACK,depth=3)
    agent2 = RandomAI(Reversi.WHITE)  # Minimax代理
    agent1_wins, agent2_wins, draws = test_agents(agent1, agent2, Reversi, games=100)



