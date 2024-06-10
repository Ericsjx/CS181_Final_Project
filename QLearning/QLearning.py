from game import Reversi
import random
import numpy as np
import pickle
from collections import defaultdict

class QLearningAI:
    def __init__(self, player, alpha=0.1, gamma=0.9, epsilon=0.1, default_q=0.0):
        self.player = player
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.default_q = default_q
        self.q_table = defaultdict(lambda: default_q)

    def get_state(self, game):
        return tuple(map(tuple, game.board))

    def get_q_value(self, state, action):
        return self.q_table[(state, action)]

    def set_q_value(self, state, action, value):
        self.q_table[(state, action)] = value

    def update_q_value(self, state, action, reward, next_state, game):
        best_next_action = self.select_best_action(next_state, game)
        if best_next_action is None:
            best_next_q_value = 0  # 如果没有有效动作，则Q值为0
        else:
            best_next_q_value = self.get_q_value(next_state, best_next_action)
        td_target = reward + self.gamma * best_next_q_value
        td_delta = td_target - self.get_q_value(state, action)
        new_value = self.get_q_value(state, action) + self.alpha * td_delta
        self.set_q_value(state, action, new_value)

    def select_best_action(self, state, game):
        valid_moves = game.get_valid_moves(self.player)
        if not valid_moves:
            return None
        q_values = [self.get_q_value(state, move) for move in valid_moves]
        max_q = max(q_values)
        best_moves = [move for move, q in zip(valid_moves, q_values) if q == max_q]
        return random.choice(best_moves)

    def select_move(self, game):
        state = self.get_state(game)
        if random.random() < self.epsilon:
            return random.choice(game.get_valid_moves(self.player))
        else:
            return self.select_best_action(state, game)

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = defaultdict(lambda: self.default_q, pickle.load(f))

def epsilon_greedy(player, game, epsilon):
    valid_moves = game.get_valid_moves(player.player)
    if not valid_moves:
        return None
    if random.random() < epsilon:
        return random.choice(valid_moves)
    else:
        return player.select_move(game)

def calculate_reward(game, player):
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
    reward = 0
    board = game.board
    for i in range(8):
        for j in range(8):
            if board[i][j] == player.player:
                reward += weight_matrix[i][j]
                #print(f"player in {i}{j}")
            elif board[i][j] == 3 - player.player:  # Assuming 1 and 2 are the two players
                #print(f"opponent in {i}{j}")
                reward -= weight_matrix[i][j]
    #print(f"run calculate_reward {reward}")
    return reward

def get_reward(game, agent, opponent, reward_win, reward_loss, reward_draw):
    if game.is_game_over():
        winner = game.get_winner()
        if winner == agent.player:
            return reward_win
        elif winner == opponent.player:
            return reward_loss
        else:
            return reward_draw
    return calculate_reward(game, agent)  # 根据棋盘上的棋子位置返回奖励

def get_final_reward(winner, agent, opponent, reward_win, reward_loss, reward_draw):
    if winner:
        return reward_win if winner == agent.player else reward_loss
    else:
        return reward_draw

def train_agent_QLearning(episodes, agent, opponent, game_class, reward_win=1.0, reward_loss=-1.0, reward_draw=0.0, log_interval=100):
    try:
        agent.load_q_table('q_table.pkl')
        print(f"从 q_table.pkl 加载 Q 表成功")
    except FileNotFoundError:
        print(f"未找到 q_table.pkl 文件。开始使用空的 Q 表。")



    alpha_schedule = [0.8] * 1000 + [0.5] * 1000 + [0.2] * 1000 + [0.1] * 1000
    alpha_index = 0

    gamma_schedule = [0.8] * 1000 + [0.5] * 1000 + [0.2] * 1000 + [0.1] * 1000
    gamma_index = 0

    epsilon = 0.5

    for episode in range(episodes):
        game = game_class()
        state = agent.get_state(game)
        total_reward = 0

        alpha = alpha_schedule[min(episode, len(alpha_schedule) - 1)]
        gamma = gamma_schedule[min(episode, len(gamma_schedule) - 1)]

        while not game.is_game_over():
            current_player = agent if game.current_player == agent.player else opponent
            move = epsilon_greedy(current_player, game, epsilon)
            if move is None:
                break
            game.make_move(move[0], move[1], current_player.player)
            new_state = agent.get_state(game)
            reward = get_reward(game, agent, opponent, reward_win, reward_loss, reward_draw)
            agent.update_q_value(state, move, reward, new_state, game)
            state = new_state
            total_reward += reward
            epsilon = 0.5

            if game.is_game_over():
                winner = game.get_winner()
                final_reward = get_final_reward(winner, agent, opponent, reward_win, reward_loss, reward_draw)
                agent.update_q_value(state, move, final_reward, new_state, game)
                total_reward += final_reward
                break

            if not game.get_valid_moves(game.current_player):
                game.current_player = Reversi.BLACK if game.current_player == Reversi.WHITE else Reversi.WHITE

        if episode % log_interval == 0:
            print(f"第 {episode} 次训练：总奖励 = {total_reward}, 探索率 = {epsilon}, 学习率 = {alpha}, 折扣因子 = {gamma}")
            total_reward = 0

    agent.save_q_table('q_table.pkl')
