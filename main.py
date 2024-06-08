from game import Reversi
from Greedy.Greedy import GreedyAI
from MCTS.MCTS import MCTSAgent
from Minimax.Minimax import MinimaxAI
from Random.Random import RandomAI
from test_agent import test_agents

if __name__ == '__main__':
    agent1 = RandomAI(Reversi.BLACK)
    # agent1 = MCTSAgent(Reversi.BLACK, simulations=1000)  # MCTS代理
    agent2 = RandomAI(Reversi.WHITE)  # Minimax代理

    agent1_wins, agent2_wins, draws = test_agents(agent1, agent2, Reversi, games=100)

    print(f"After 100 games:")
    print(f"Agent 1 (Greedy) wins: {agent1_wins}")
    print(f"Agent 2 (Minimax) wins: {agent2_wins}")
    print(f"Draws: {draws}")



