class Reversi:
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def __init__(self):
        self.board = [[Reversi.EMPTY] * 8 for _ in range(8)]
        self.board[3][3] = self.board[4][4] = Reversi.WHITE
        self.board[3][4] = self.board[4][3] = Reversi.BLACK
        self.current_player = Reversi.BLACK

    def copy(self):
        new_game = Reversi()
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        return new_game

    def is_valid_move(self, row, col, player):
        if self.board[row][col] != Reversi.EMPTY:
            return False
        opponent = self.get_opponent(player)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                while 0 <= r < 8 and 0 <= c < 8:
                    r += dr
                    c += dc
                    if not (0 <= r < 8 and 0 <= c < 8):
                        break
                    if self.board[r][c] == player:
                        return True
                    if self.board[r][c] == Reversi.EMPTY:
                        break
        return False

    def get_valid_moves(self, player):
        return [(r, c) for r in range(8) for c in range(8) if self.is_valid_move(r, c, player)]

    def make_move(self, row, col, player):
        if not self.is_valid_move(row, col, player):
            return False
        self.board[row][col] = player
        opponent = self.get_opponent(player)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            flip_positions = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                flip_positions.append((r, c))
                r += dr
                c += dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == player:
                for fr, fc in flip_positions:
                    self.board[fr][fc] = player
        if self.get_valid_moves(opponent):
            self.current_player = opponent
        return True

    def get_opponent(self, player):
        return Reversi.BLACK if player == Reversi.WHITE else Reversi.WHITE

    def is_game_over(self):
        return not self.get_valid_moves(Reversi.BLACK) and not self.get_valid_moves(Reversi.WHITE)

    def get_winner(self):
        black_count = sum(row.count(Reversi.BLACK) for row in self.board)
        white_count = sum(row.count(Reversi.WHITE) for row in self.board)
        if black_count > white_count:
            return Reversi.BLACK
        elif white_count > black_count:
            return Reversi.WHITE
        else:
            return Reversi.EMPTY

    def print_board(self):
        for row in self.board:
            print(' '.join(['.', 'B', 'W'][cell] for cell in row))