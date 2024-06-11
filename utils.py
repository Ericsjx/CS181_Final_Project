from game import Reversi
import tkinter as tk
from tkinter import messagebox


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

    def reset(self, game, agent1, agent2):
        self.game = game
        self.agent1 = agent1
        self.agent2 = agent2
        self.update_board() 

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