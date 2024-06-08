import tkinter as tk
from tkinter import messagebox
import random
import copy
import math
import numpy as np
import pickle
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.optimizers import Adam

class Reversi:
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def __init__(self):
        self.board = [[Reversi.EMPTY] * 8 for _ in range(8)]
        self.board[3][3] = self.board[4][4] = Reversi.WHITE
        self.board[3][4] = self.board[4][3] = Reversi.BLACK
        self.current_player = Reversi.BLACK

    def is_valid_move(self, row, col, player):
        if self.board[row][col] != Reversi.EMPTY:
            return False
        opponent = Reversi.BLACK if player == Reversi.WHITE else Reversi.WHITE
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
        opponent = Reversi.BLACK if player == Reversi.WHITE else Reversi.WHITE
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
        # 只有在对方有合法移动时才切换当前玩家
        if self.get_valid_moves(opponent):
            self.current_player = opponent
        return True

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

class RandomAI:
    def __init__(self, player):
        self.player = player

    def select_move(self, game):
        moves = game.get_valid_moves(self.player)
        if not moves:
            return None
        return random.choice(moves)

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

class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = copy.deepcopy(game)
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = game.get_valid_moves(game.current_player)

    def select_child(self):
        return sorted(self.children, key=lambda c: c.wins / c.visits + math.sqrt(2 * math.log(self.visits) / c.visits))[-1]

    def add_child(self, move, game):
        child_node = MCTSNode(game, self, move)
        self.untried_moves.remove(move)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

class MCTS:
    def __init__(self, player, iterations=1000):
        self.player = player
        self.iterations = iterations

    def select_move(self, game):
        root = MCTSNode(game)
        for _ in range(self.iterations):
            node = root
            game_copy = copy.deepcopy(game)

            # Selection
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
                game_copy.make_move(node.move[0], node.move[1], game_copy.current_player)

            # Expansion
            if node.untried_moves != []:
                move = random.choice(node.untried_moves)
                game_copy.make_move(move[0], move[1], game_copy.current_player)
                node = node.add_child(move, game_copy)

            # Simulation
            while game_copy.get_valid_moves(game_copy.current_player):
                move = random.choice(game_copy.get_valid_moves(game_copy.current_player))
                game_copy.make_move(move[0], move[1], game_copy.current_player)

            # Backpropagation
            result = self.get_result(game_copy)
            while node is not None:
                node.update(result)
                node = node.parent

        return sorted(root.children, key=lambda c: c.visits)[-1].move

    def get_result(self, game):
        winner = game.get_winner()
        if winner == self.player:
            return 1
        elif winner == Reversi.EMPTY:
            return 0.5
        else:
            return 0

class MLAgent:
    def __init__(self, player):
        self.player = player
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Flatten(input_shape=(8, 8)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(), loss='mse')
        return model

    def select_move(self, game):
        valid_moves = game.get_valid_moves(self.player)
        if not valid_moves:
            return None
        best_move = None
        best_value = float('-inf')
        for move in valid_moves:
            game_copy = copy.deepcopy(game)
            game_copy.make_move(move[0], move[1], self.player)
            board = np.array(game_copy.board).reshape((1, 8, 8))
            value = self.model.predict(board)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

class ReversiGUI:
    def __init__(self, root, player1, player2):
        self.root = root
        self.player1 = player1
        self.player2 = player2
        self.game = Reversi()
        self.board_frame = tk.Frame(root)
        self.board_frame.pack()
        self.cells = [[None] * 8 for _ in range(8)]
        self.create_board()
        self.update_board()
        self.root.after(1000, self.play)

    def create_board(self):
        for r in range(8):
            for c in range(8):
                self.cells[r][c] = tk.Button(self.board_frame, width=4, height=2,
                                             command=lambda row=r, col=c: self.make_move(row, col))
                self.cells[r][c].grid(row=r, column=c)

    def update_board(self):
        for r in range(8):
            for c in range(8):
                cell_value = self.game.board[r][c]
                if cell_value == Reversi.BLACK:
                    self.cells[r][c].config(text='B', bg='black', fg='white')
                elif cell_value == Reversi.WHITE:
                    self.cells[r][c].config(text='W', bg='white', fg='black')
                else:
                    self.cells[r][c].config(text='', bg='green')

    def make_move(self, row, col):
        if self.game.make_move(row, col, self.game.current_player):
            self.update_board()
            if self.game.is_game_over():
                self.show_winner()
            else:
                self.root.after(1000, self.play)

    def play(self):
        if self.game.current_player == self.player1.player:
            move = self.player1.select_move(self.game)
        else:
            move = self.player2.select_move(self.game)
        if move:
            self.make_move(move[0], move[1])

    def show_winner(self):
        winner = self.game.get_winner()
        if winner == Reversi.BLACK:
            messagebox.showinfo("Game Over", "Black wins!")
        elif winner == Reversi.WHITE:
            messagebox.showinfo("Game Over", "White wins!")
        else:
            messagebox.showinfo("Game Over", "It's a tie!")

def train_agent(episodes, agent, opponent, game_class, reward_win=1.0, reward_loss=-1.0, reward_draw=0.0):
    for _ in range(episodes):
        game = game_class()
        while not game.is_game_over():
            current_player = agent if game.current_player == agent.player else opponent
            state = copy.deepcopy(game.board)
            move = current_player.select_move(game)
            if move:
                game.make_move(move[0], move[1], current_player.player)
            # 如果当前玩家无法移动，交替到另一玩家
            if not game.get_valid_moves(game.current_player):
                game.current_player = Reversi.BLACK if game.current_player == Reversi.WHITE else Reversi.WHITE
            new_state = copy.deepcopy(game.board)
            reward = 0
            if game.is_game_over():
                winner = game.get_winner()
                if winner == agent.player:
                    reward = reward_win
                elif winner == opponent.player:
                    reward = reward_loss
                else:
                    reward = reward_draw
            agent.update_q_value(state, move, reward, new_state)
    agent.save_q_table('q_table.pkl')

def test_agent(agent, opponent, game_class, num_games=100):
    wins = 0
    losses = 0
    draws = 0
    for _ in range(num_games):
        game = game_class()
        while not game.is_game_over():
            current_player = agent if game.current_player == agent.player else opponent
            move = current_player.select_move(game)
            if move:
                game.make_move(move[0], move[1], current_player.player)
            # 如果当前玩家无法移动，交替到另一玩家
            if not game.get_valid_moves(game.current_player):
                game.current_player = Reversi.BLACK if game.current_player == Reversi.WHITE else Reversi.WHITE
        winner = game.get_winner()
        if winner == agent.player:
            wins += 1
        elif winner == opponent.player:
            losses += 1
        else:
            draws += 1
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Reversi")
    player1 = RandomAI(Reversi.BLACK)
    player2 = MinimaxAI(Reversi.WHITE)
    gui = ReversiGUI(root, player1, player2)
    root.mainloop()


# class QLearningAgent:
#     def __init__(self, player, alpha=0.1, gamma=0.9, epsilon=0.1):
#         self.player = player
#         self.alpha = alpha  # 学习率
#         self.gamma = gamma  # 折扣因子
#         self.epsilon = epsilon  # 探索概率
#         self.q_table = {}  # Q表

#     def get_q_value(self, state, action):
#         # 将状态和动作转化为字符串作为字典的键
#         state_action = (str(state), action)
#         return self.q_table.get(state_action, 0.0)

#     def set_q_value(self, state, action, value):
#         state_action = (str(state), action)
#         self.q_table[state_action] = value

#     def select_move(self, game):
#         valid_moves = game.get_valid_moves(self.player)
#         if not valid_moves:
#             return None

#         # epsilon-greedy策略
#         if random.uniform(0, 1) < self.epsilon:
#             return random.choice(valid_moves)  # 探索：随机选择一个动作

#         # 利用：选择Q值最高的动作
#         q_values = [self.get_q_value(game.board, move) for move in valid_moves]
#         max_q = max(q_values)
#         best_moves = [move for move, q_value in zip(valid_moves, q_values) if q_value == max_q]
#         return random.choice(best_moves)

#     def update_q_value(self, old_state, action, reward, new_state):
#         old_q_value = self.get_q_value(old_state, action)
#         future_rewards = [self.get_q_value(new_state, a) for a in game.get_valid_moves(self.player)]
#         max_future_q_value = max(future_rewards) if future_rewards else 0
#         new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_q_value - old_q_value)
#         self.set_q_value(old_state, action, new_q_value)

#     def save_q_table(self, filename):
#         with open(filename, 'wb') as f:
#             pickle.dump(self.q_table, f)

#     def load_q_table(self, filename):
#         with open(filename, 'rb') as f:
#             self.q_table = pickle.load(f)

# def train_agent(episodes, agent, opponent, game_class, reward_win=1.0, reward_loss=-1.0, reward_draw=0.0):
#     for _ in range(episodes):
#         game = game_class()
#         while not game.is_game_over():
#             current_player = agent if game.current_player == agent.player else opponent
#             state = copy.deepcopy(game.board)
#             move = current_player.select_move(game)
#             if move:
#                 game.make_move(move[0], move[1], current_player.player)
#             new_state = copy.deepcopy(game.board)
#             reward = 0
#             if game.is_game_over():
#                 winner = game.get_winner()
#                 if winner == agent.player:
#                     reward = reward_win
#                 elif winner == opponent.player:
#                     reward = reward_loss
#                 else:
#                     reward = reward_draw
#             agent.update_q_value(state, move, reward, new_state)
#     agent.save_q_table('q_table.pkl')

# def test_agent(agent, opponent, game_class, num_games=100):
#     wins = 0
#     losses = 0
#     draws = 0
#     for _ in range(num_games):
#         game = game_class()
#         while not game.is_game_over():
#             current_player = agent if game.current_player == agent.player else opponent
#             move = current_player.select_move(game)
#             if move:
#                 game.make_move(move[0], move[1], current_player.player)
#         winner = game.get_winner()
#         if winner == agent.player:
#             wins += 1
#         elif winner == opponent.player:
#             losses += 1
#         else:
#             draws += 1
#     print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")

# if __name__ == '__main__':
#     player1 = QLearningAgent(Reversi.BLACK)
#     player2 = RandomAI(Reversi.WHITE)
#     train_agent(10000, player1, player2, Reversi)
#     player1.load_q_table('q_table.pkl')
#     test_agent(player1, player2, Reversi)

