from game import Reversi

class GreedyAI:
    def __init__(self, player):
        self.player = player

    def select_move(self, game):
        max_flips = -1
        best_move = None
        for move in game.get_valid_moves(self.player):
            flips = self.count_flips(game, move[0], move[1])
            if flips > max_flips:
                max_flips = flips
                best_move = move
        return best_move

    def count_flips(self, game, row, col):
        opponent = Reversi.BLACK if self.player == Reversi.WHITE else Reversi.WHITE
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        flip_count = 0
        for dr, dc in directions:
            r, c = row + dr, col + dc
            count = 0
            while 0 <= r < 8 and 0 <= c < 8 and game.board[r][c] == opponent:
                count += 1
                r += dr
                c += dc
            if 0 <= r < 8 and 0 <= c < 8 and game.board[r][c] == self.player:
                flip_count += count
        return flip_count