"""
Tic-tac-toe game engine (pure Python, no hardware).

The board is a list of 9 cells, same symbols as the vision code:
    "X", "O", or "-" (empty)
Cell numbers:   0 1 2
                3 4 5
                6 7 8

Usage:
    python game.py play       # you (X) play against the AI (O) in the terminal
    python game.py test       # AI vs random player, 1000 games each way
    python game.py compare    # check plain minimax and alpha-beta agree
"""

import sys
import random

EASY_CHANCE = 0.0   # 0.0 = perfect AI. Set 0.3 for easy mode (30% random moves)

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),      # rows
         (0, 3, 6), (1, 4, 7), (2, 5, 8),      # columns
         (0, 4, 8), (2, 4, 6)]                 # diagonals


# ---------------------------------------------------------------- basics
def winner(board):
    """Return "X" or "O" if someone has three in a row, else None."""
    for a, b, c in LINES:
        if board[a] != "-" and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_full(board):
    return "-" not in board


def empty_cells(board):
    return [i for i in range(9) if board[i] == "-"]


def other(player):
    return "O" if player == "X" else "X"


def print_board(board):
    for r in range(3):
        print(" ".join(board[r * 3:r * 3 + 3]))
    print()


# ---------------------------------------------------------------- minimax
def minimax(board, player, ai):
    """
    Score the board from the AI's point of view:
        +1 = AI wins, 0 = draw, -1 = AI loses.
    `player` is whose turn it is now.
    """
    w = winner(board)
    if w == ai:
        return 1
    if w is not None:
        return -1
    if is_full(board):
        return 0

    scores = []
    for cell in empty_cells(board):
        board[cell] = player                              # try the move
        scores.append(minimax(board, other(player), ai))
        board[cell] = "-"                                 # undo it

    return max(scores) if player == ai else min(scores)   # AI maximizes, opponent minimizes


def minimax_ab(board, player, ai, alpha=-2, beta=2):
    """Same as minimax, but skips branches that cannot change the answer (faster)."""
    w = winner(board)
    if w == ai:
        return 1
    if w is not None:
        return -1
    if is_full(board):
        return 0

    if player == ai:
        best = -2
        for cell in empty_cells(board):
            board[cell] = player
            best = max(best, minimax_ab(board, other(player), ai, alpha, beta))
            board[cell] = "-"
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = 2
        for cell in empty_cells(board):
            board[cell] = player
            best = min(best, minimax_ab(board, other(player), ai, alpha, beta))
            board[cell] = "-"
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


def best_move(board, ai, use_pruning=True, easy_chance=None):
    """Return the cell number (0-8) the AI should play."""
    if easy_chance is None:
        easy_chance = EASY_CHANCE
    cells = empty_cells(board)

    if random.random() < easy_chance:                     # easy mode: random move
        return random.choice(cells)

    score_fn = minimax_ab if use_pruning else minimax
    best_cell, best_score = None, -2
    for cell in cells:
        board[cell] = ai
        score = score_fn(board, other(ai), ai)
        board[cell] = "-"
        if score > best_score:
            best_cell, best_score = cell, score
    return best_cell


# ---------------------------------------------------------------- 1000-game test
def play_one_game(ai, ai_first, easy_chance=0.0):
    """AI vs a random player. Returns "ai", "random" or "draw"."""
    board = ["-"] * 9
    human = other(ai)
    turn = ai if ai_first else human
    while True:
        if turn == ai:
            move = best_move(board, ai, easy_chance=easy_chance)
        else:
            move = random.choice(empty_cells(board))
        board[move] = turn
        w = winner(board)
        if w == ai:
            return "ai"
        if w is not None:
            return "random"
        if is_full(board):
            return "draw"
        turn = other(turn)


def run_test(games=1000, easy_chance=0.0):
    lines = []
    for ai_first in (True, False):
        results = {"ai": 0, "draw": 0, "random": 0}
        for _ in range(games):
            results[play_one_game("O", ai_first, easy_chance)] += 1
        who = "AI goes first" if ai_first else "AI goes second"
        lines.append(f"{who}: {games} games -> "
                     f"AI wins {results['ai']}, draws {results['draw']}, "
                     f"AI losses {results['random']}")
    for line in lines:
        print(line)
    with open("results.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved to results.txt")


# ---------------------------------------------------------------- compare
def run_compare(boards=300):
    """Play random partial games and check both versions give the same scores."""
    random.seed(1)
    checked = 0
    for _ in range(boards):
        board = ["-"] * 9
        player = random.choice("XO")
        for _ in range(random.randint(0, 5)):             # random start position
            if winner(board) or is_full(board):
                break
            board[random.choice(empty_cells(board))] = player
            player = other(player)
        if winner(board) or is_full(board):
            continue
        plain = minimax(board, player, "O")
        fast = minimax_ab(board, player, "O")
        if plain != fast:
            print("MISMATCH on board:", board, plain, fast)
            return
        checked += 1
    print(f"Plain minimax and alpha-beta agree on {checked} boards.")


# ---------------------------------------------------------------- play in terminal
def play():
    board = ["-"] * 9
    me, ai = "X", "O"
    print("You are X. Enter a cell number 1-9:\n 1 2 3\n 4 5 6\n 7 8 9\n")
    turn = me
    while True:
        print_board(board)
        if turn == me:
            try:
                move = int(input("Your move (1-9): ")) - 1
            except ValueError:
                print("Please type a number from 1 to 9.")
                continue
            if move not in range(9) or board[move] != "-":
                print("That cell is not available.")
                continue
        else:
            move = best_move(board, ai)
            print(f"AI plays cell {move + 1}")
        board[move] = turn
        w = winner(board)
        if w or is_full(board):
            print_board(board)
            print("You win!" if w == me else "AI wins!" if w == ai else "Draw.")
            return
        turn = other(turn)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "play":
        play()
    elif cmd == "test":
        run_test()
    elif cmd == "compare":
        run_compare()
    else:
        print(__doc__)