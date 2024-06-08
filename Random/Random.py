from game import Reversi
import random

class RandomAI:
    def __init__(self, player):
        self.player = player

    def select_move(self, game):
        moves = game.get_valid_moves(self.player)
        if not moves:
            return None
        return random.choice(moves)