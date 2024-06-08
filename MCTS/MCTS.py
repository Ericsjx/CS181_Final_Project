from game import Reversi
import math
import random 
import copy

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_valid_moves(self.state.current_player))

    def best_child(self, exploration_weight=1.4):
        # Prevent division by zero by adding a small epsilon
        epsilon = 1e-6
        choices_weights = [
            (child.wins / (child.visits + epsilon)) + exploration_weight * math.sqrt((2 * math.log(self.visits + 1) / (child.visits + epsilon)))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

class MCTSAgent:
    def __init__(self, player, simulations=1000):
        self.player = player
        self.simulations = simulations

    def select_move(self, game):
        root = MCTSNode(state=game)
        for _ in range(self.simulations):
            node = root
            state = game.copy()

            # Selection
            while node.children:
                node = node.best_child()
                state.make_move(node.move[0], node.move[1], state.current_player)

            # Expansion
            if not state.is_game_over() and not node.is_fully_expanded():
                valid_moves = state.get_valid_moves(state.current_player)
                for move in valid_moves:
                    if not any(child.move == move for child in node.children):
                        state.make_move(move[0], move[1], state.current_player)
                        new_node = MCTSNode(state=state.copy(), parent=node, move=move)
                        node.children.append(new_node)
                        break

            # Simulation
            simulation_state = state.copy()
            while not simulation_state.is_game_over():
                moves = simulation_state.get_valid_moves(simulation_state.current_player)
                if moves:
                    move = random.choice(moves)
                    simulation_state.make_move(move[0], move[1], simulation_state.current_player)
                else:
                    simulation_state.current_player = Reversi.BLACK if simulation_state.current_player == Reversi.WHITE else Reversi.WHITE

            # Backpropagation
            reward = 1 if simulation_state.get_winner() == self.player else 0
            while node:
                node.visits += 1
                node.wins += reward
                node = node.parent

        return root.best_child(exploration_weight=0).move

    def simulate(self, state):
        while not state.is_game_over():
            moves = state.get_valid_moves(state.current_player)
            if moves:
                move = random.choice(moves)
                state.make_move(move[0], move[1], state.current_player)
            else:
                state.current_player = Reversi.BLACK if state.current_player == Reversi.WHITE else Reversi.WHITE
        return 1 if state.get_winner() == self.player else 0