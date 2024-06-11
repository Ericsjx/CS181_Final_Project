from game import Reversi
import math
import random 

class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.wins = 0
        self.visits = 0
        self.children = []
        self.untried_moves = game.get_valid_moves(game.current_player)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.41):
        choices_weights = [
            (child.wins / child.visits) + exploration_weight * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        move = self.untried_moves.pop()
        new_game = self.game.copy()
        new_game.make_move(move[0], move[1], new_game.current_player)
        child_node = MCTSNode(new_game, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def best_move(self):
        return max(self.children, key=lambda child: child.visits).move

class MCTSAgent:
    def __init__(self, player, simulations=1000):
        self.player = player
        self.simulations = simulations

    def select_move(self, game):
        root = MCTSNode(game)
        
        for i in range(self.simulations):
            node = root
            game_copy = game.copy()
            
            # Print simulation number every 10 simulations
            # if (i + 1) % 10 == 0:
            #     print(f"Simulation {i+1}/{self.simulations}")
            
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                game_copy.make_move(node.move[0], node.move[1], game_copy.current_player)
            
            # Expansion
            if not node.is_fully_expanded():
                node = node.expand()
                game_copy.make_move(node.move[0], node.move[1], game_copy.current_player)
            
            # Simulation
            while not game_copy.is_game_over():
                possible_moves = game_copy.get_valid_moves(game_copy.current_player)
                if possible_moves:
                    move = random.choice(possible_moves)
                    game_copy.make_move(move[0], move[1], game_copy.current_player)
                else:
                    game_copy.current_player = game_copy.get_opponent(game_copy.current_player)
            
            # Backpropagation
            winner = game_copy.get_winner()
            while node is not None:
                node.visits += 1
                if node.game.current_player != self.player:  # If the move to get to this node was ours
                    if winner == self.player:
                        node.wins += 1
                    elif winner != Reversi.EMPTY:
                        node.wins -= 1
                node = node.parent
        
        best_move = root.best_move()
        # print(f"Best Move chosen: {best_move}")
        return best_move