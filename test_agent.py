from game import Reversi

def play_game(agent1, agent2, game_class, verbose=False):
    game = game_class()
    current_agent = agent1

    while not game.is_game_over():
        valid_moves = game.get_valid_moves(game.current_player)
        if valid_moves:
            move = current_agent.select_move(game)
            if move is not None:
                game.make_move(move[0], move[1], game.current_player)
        else:
            # 如果没有合法移动，切换当前玩家
            game.current_player = Reversi.BLACK if game.current_player == Reversi.WHITE else Reversi.WHITE

        current_agent = agent1 if game.current_player == agent1.player else agent2

    winner = game.get_winner()
    if verbose:
        if winner == agent1.player:
            print(f"Agent 1 ({agent1.__class__.__name__}) wins!")
        elif winner == agent2.player:
            print(f"Agent 2 ({agent2.__class__.__name__}) wins!")
        else:
            print("Draw!")

    return winner



def test_agents(agent1, agent2, game_class, games=100):
    agent1_wins = 0
    agent2_wins = 0
    draws = 0

    for i in range(games):
        print(f"Game {i+1}:")
        winner = play_game(agent1, agent2, game_class, verbose=True)
        if winner == agent1.player:
            agent1_wins += 1
        elif winner == agent2.player:
            agent2_wins += 1
        else:
            draws += 1

        # 交换起始玩家
        agent1, agent2 = agent2, agent1
        agent1_wins, agent2_wins = agent2_wins, agent1_wins
        # agent1.reset()
        # agent2.reset()

    print(f"Agent 1 ({agent1.__class__.__name__}) wins: {agent1_wins}")
    print(f"Agent 2 ({agent2.__class__.__name__}) wins: {agent2_wins}")
    print(f"Draws: {draws}")

    return agent1_wins, agent2_wins, draws