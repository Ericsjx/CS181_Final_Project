class MinimaxAI:
    def __init__(self, player, depth=3):
        self.player = player
        self.depth = depth

    def select_move(self, game):
        def minimax(game, depth, alpha, beta, maximizing_player):
            if depth == 0 or game.is_game_over():
                return self.evaluate_board(game), None

            valid_moves = game.get_valid_moves(self.player if maximizing_player else game.get_opponent(self.player))

            if not valid_moves:
                return self.evaluate_board(game), None

            if maximizing_player:
                max_eval = float('-inf')
                best_move = None
                for move in valid_moves:
                    new_game = game.copy()
                    new_game.make_move(move[0], move[1], self.player)
                    eval, _ = minimax(new_game, depth - 1, alpha, beta, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break  # Beta剪枝
                return max_eval, best_move
            else:
                min_eval = float('inf')
                best_move = None
                opponent = game.get_opponent(self.player)
                for move in valid_moves:
                    new_game = game.copy()
                    new_game.make_move(move[0], move[1], opponent)
                    eval, _ = minimax(new_game, depth - 1, alpha, beta, True)
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break  # Alpha剪枝
                return min_eval, best_move

        _, best_move = minimax(game, self.depth, float('-inf'), float('inf'), True)

        if best_move is None:
            valid_moves = game.get_valid_moves(self.player)
            if valid_moves:
                best_move = valid_moves[0]  
            else:
                return None  

        return best_move

    def evaluate_board(self, game):

        weight_matrix = [
            [100, -20, 10, 5, 5, 10, -20, 100],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [10, -2, -1, -1, -1, -1, -2, 10],
            [5, -2, -1, -1, -1, -1, -2, 5],
            [5, -2, -1, -1, -1, -1, -2, 5],
            [10, -2, -1, -1, -1, -1, -2, 10],
            [-20, -50, -2, -2, -2, -2, -50, -20],
            [100, -20, 10, 5, 5, 10, -20, 100]
        ]

        player_score = 0
        opponent_score = 0
        opponent = game.get_opponent(self.player)

        for r in range(8):
            for c in range(8):
                if game.board[r][c] == self.player:
                    player_score += weight_matrix[r][c]
                elif game.board[r][c] == opponent:
                    opponent_score += weight_matrix[r][c]

        return player_score - opponent_score
