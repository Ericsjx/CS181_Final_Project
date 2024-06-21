import numpy as np
import pickle
import random
from game import Reversi

class ApproximateQLearningAI:
    def __init__(self, player, alpha=0.1, gamma=0.9, epsilon=0.1, feature_size=133):
        self.player = player
        self.alpha = alpha 
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.weights = np.zeros(feature_size)  

    def get_state(self, game):
        return tuple(map(tuple, game.board))

    def get_features(self, state, action, game):
        features = np.zeros(133)
        board = np.array(state)
        i, j = action
        features[i * 8 + j] = 1


        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for corner in corners:
            if board[corner[0]][corner[1]] == self.player:
                features[64 + corners.index(corner)] = 1


        stability = self.calculate_stability(board)
        features[68:132] = stability.flatten()


        valid_moves = len(game.get_valid_moves(self.player))
        features[132] = valid_moves


        if np.linalg.norm(features) != 0:
            features = features / np.linalg.norm(features)

        return features

    def calculate_stability(self, board):
        stability = np.zeros((8, 8))

        for i in range(8):
            for j in range(8):
                if board[i][j] == self.player:
                    if i == 0 or i == 7 or j == 0 or j == 7:
                        stability[i][j] = 1
        return stability

    def get_q_value(self, state, action, game):
        features = self.get_features(state, action, game)
        return np.dot(self.weights, features)

    def update_weights(self, state, action, reward, next_state, game):
        features = self.get_features(state, action, game)
        best_next_action = self.select_best_action(next_state, game)
        if best_next_action is None:
            best_next_q_value = 0  
        else:
            best_next_q_value = self.get_q_value(next_state, best_next_action, game)
        current_q_value = self.get_q_value(state, action, game)
        td_target = reward + self.gamma * best_next_q_value
        td_delta = td_target - current_q_value
        if not np.isnan(td_delta) and not np.isnan(features).any():  
            self.weights += self.alpha * td_delta * features

    def select_best_action(self, state, game):
        valid_moves = game.get_valid_moves(self.player)
        if not valid_moves:
            return None
        q_values = [self.get_q_value(state, move, game) for move in valid_moves]
        max_q = max(q_values)
        best_moves = [move for move, q in zip(valid_moves, q_values) if q == max_q]
        if not best_moves:
            return None
        return random.choice(best_moves)

    def select_move(self, game):
        state = self.get_state(game)
        if random.random() < self.epsilon:
            valid_moves = game.get_valid_moves(self.player)
            if not valid_moves:
                return None
            return random.choice(valid_moves)
        else:
            return self.select_best_action(state, game)

    def save_weights(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)

    def load_weights(self, filename):
        with open(filename, 'rb') as f:
            self.weights = pickle.load(f)

    def update_parameters(self, episode, initial_epsilon=1.0, min_epsilon=0.01, decay_rate=0.995):
        self.epsilon = max(min_epsilon, initial_epsilon * (decay_rate ** episode))
        alpha_schedule = [0.8] * 1000 + [0.5] * 1000 + [0.2] * 1000 + [0.1] * 1000
        gamma_schedule = [0.8] * 1000 + [0.5] * 1000 + [0.2] * 1000 + [0.1] * 1000
        self.alpha = alpha_schedule[min(episode, len(alpha_schedule) - 1)]
        self.gamma = gamma_schedule[min(episode, len(gamma_schedule) - 1)]

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
            elif board[i][j] == 3 - player.player:  # Assuming 1 and 2 are the two players
                reward -= weight_matrix[i][j]
    
    # 增加边角占领奖励
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    for corner in corners:
        if board[corner[0]][corner[1]] == player.player:
            reward += 50

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
    return calculate_reward(game, agent)  

def get_final_reward(winner, agent, opponent, reward_win, reward_loss, reward_draw):
    if winner:
        return reward_win if winner == agent.player else reward_loss
    else:
        return reward_draw

def train_agent_ApproximateQLearning(episodes, agent, opponent, game_class, reward_win=1.0, reward_loss=-1.0, reward_draw=0.0, log_interval=100):
    try:
        agent.load_weights('weights.pkl')
        print(f"从 weights.pkl 加载权重成功")
    except FileNotFoundError:
        print(f"未找到 weights.pkl 文件。开始使用初始权重。")

    total_rewards = []  # 记录每次训练的总奖励
    win_rate = []  # 记录每次训练的胜率

    for episode in range(episodes):
        game = game_class()
        state = agent.get_state(game)
        total_reward = 0
        wins = 0

        agent.update_parameters(episode)

        while not game.is_game_over():
            current_player = agent if game.current_player == agent.player else opponent
            move = epsilon_greedy(current_player, game, agent.epsilon)
            if move is None:
                break
            game.make_move(move[0], move[1], current_player.player)
            new_state = agent.get_state(game)
            reward = get_reward(game, agent, opponent, reward_win, reward_loss, reward_draw)
            agent.update_weights(state, move, reward, new_state, game)
            state = new_state
            total_reward += reward

            if game.is_game_over():
                winner = game.get_winner()
                final_reward = get_final_reward(winner, agent, opponent, reward_win, reward_loss, reward_draw)
                agent.update_weights(state, move, final_reward, new_state, game)
                total_reward += final_reward
                if winner == agent.player:
                    wins += 1
                break

            if not game.get_valid_moves(game.current_player):
                game.current_player = Reversi.BLACK if game.current_player == Reversi.WHITE else Reversi.WHITE

        total_rewards.append(total_reward)
        win_rate.append(wins)

        if episode % log_interval == 0:
            avg_reward = np.mean(total_rewards[-log_interval:])
            win_percentage = np.mean(win_rate[-log_interval:]) * 100
            print(f"第 {episode} 次训练：平均奖励 = {avg_reward:.2f}, 胜率 = {win_percentage:.2f}%, 探索率 = {agent.epsilon}, 学习率 = {agent.alpha}, 折扣因子 = {agent.gamma}")
            total_reward = 0

    agent.save_weights('weights.pkl')
    return total_rewards, win_rate  # 返回训练的奖励和胜率数据
