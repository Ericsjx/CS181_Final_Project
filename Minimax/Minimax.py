from game import Reversi
import copy

class MinimaxAI:
    def __init__(self, player, depth=3):
        self.player = player
        self.depth = depth

    def select_move(self, game):
        def minimax(game, depth, maximizing_player):
            if depth == 0 or game.is_game_over():
                return self.evaluate_board(game)
            if maximizing_player:
                max_eval = float('-inf')
                for move in game.get_valid_moves(self.player):
                    new_game = copy.deepcopy(game)
                    new_game.make_move(move[0], move[1], self.player)
                    eval = minimax(new_game, depth - 1, False)
                    max_eval = max(max_eval, eval)
                return max_eval
            else:
                min_eval = float('inf')
                opponent = Reversi.BLACK if self.player == Reversi.WHITE else Reversi.WHITE
                for move in game.get_valid_moves(opponent):
                    new_game = copy.deepcopy(game)
                    new_game.make_move(move[0], move[1], opponent)
                    eval = minimax(new_game, depth - 1, True)
                    min_eval = min(min_eval, eval)
                return min_eval

        best_move = None
        best_value = float('-inf')
        for move in game.get_valid_moves(self.player):
            new_game = copy.deepcopy(game)
            new_game.make_move(move[0], move[1], self.player)
            move_value = minimax(new_game, self.depth - 1, False)
            if move_value > best_value:
                best_value = move_value
                best_move = move
        return best_move

    def evaluate_board(self, game):
        player_count = sum(row.count(self.player) for row in game.board)
        opponent = Reversi.BLACK if self.player == Reversi.WHITE else Reversi.WHITE
        opponent_count = sum(row.count(opponent) for row in game.board)
        return player_count - opponent_count