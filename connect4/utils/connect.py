import numpy as np
from numba import njit

SEVEN = 7
SIX = 6
N = 4

MONTECARLO_SAMPLES = 20


# Board can be initialized with `board = np.zeros((NUM_COLUMNS, COLUMN_SIX), dtype=np.byte)`
# Notez Bien: Connect 4 "columns" are actually NumPy "rows"


@njit
def valid_moves(board: np.ndarray) -> np.ndarray:
    """Returns columns where a disc may be played"""
    return np.where(board[:, -1] == 0)[0]


@njit
def play(board: np.ndarray, column: int, player: int) -> None:
    """Updates `board` as `player` drops a disc in `column`"""
    index = np.argmin(np.abs(board[column, :]))
    board[column, index] = player


@njit
def take_back(board: np.ndarray, column: int, height: int = SIX) -> None:
    """Updates `board` removing top disc from `column`"""
    np.abs(board[column, ::-1])
    np.argmax(np.abs(board[column, ::-1]))
    index = height - np.argmax(np.abs(board[column, ::-1])) - 1
    board[column, index] = 0


@njit
def _n_in_a_row(board: np.ndarray, player_id: int, width: int = SEVEN, height: int = SIX, n: int = N) -> bool:
    array = np.array(
        [np.all(board[i:i + n, j] == player_id) for i in range(width - n + 1) for j in range(height)]
    )
    out = np.any(array)
    return out


@njit
def _n_in_a_col(board: np.ndarray, player_id: int, width: int = SEVEN, height: int = SIX, n: int = N) -> bool:
    array = np.array(
        [np.all(board[i, j:j + n] == player_id) for i in range(width) for j in range(height - n + 1)]
    )
    out = np.any(array)
    return out


@njit
def _n_in_a_diag(board: np.ndarray, player_id: int, width: int = SEVEN, height: int = SIX, n: int = N) -> bool:
    array = np.array(
        [np.all(np.diag(board[j:j + n, i:i + n]) == player_id)
         for i in range(height - n + 1) for j in range(width - n + 1)]
    )
    out = np.any(array)
    return out


@njit
def n_in_a_board(board: np.ndarray, player_id: int, width: int = SEVEN, height: int = SIX, n: int = N) -> bool:
    """Returns whether there are four in a row for the player"""
    array = np.array([
        _n_in_a_row(board, player_id, width, height, n), _n_in_a_col(board, player_id, width, height, n),
        _n_in_a_diag(board, player_id, width, height, n), _n_in_a_diag(np.flipud(board), player_id, width, height, n)
    ])
    out = np.any(array)
    return out


@njit
def _mc(board, player):
    p = -player
    moves = valid_moves(board)
    while len(moves) > 0:
        p = -p
        c = np.random.choice(moves)
        play(board, c, p)
        four = n_in_a_board(board, p)
        if four:
            return p
        moves = valid_moves(board)
    return 0


@njit
def montecarlo(board, player, samples: int = MONTECARLO_SAMPLES):
    cnt = np.array([_mc(np.copy(board), player) for _ in range(samples)])
    return np.mean(cnt)


# @njit
# def eval_board(board, player, samples=100):
#     player1 = n_in_a_board(board, 1)
#     if player1:
#         # Alice won
#         return 1
#     player2 = n_in_a_board(board, 1)
#     if player2:
#         # Bob won
#         return -1
#     # Not terminal, let's simulate...
#     simul = montecarlo(board, player, samples=samples)
#     return simul


def print_board(board):
    symbol1, symbol2 = 'X', 'O'
    pre = '        '

    print()
    print()
    print()
    for i in range(SIX - 1, -1, -1):
        line = []
        for j in range(SEVEN):
            if board[j][i] == 1:
                line.append(symbol1)
            elif board[j][i] == -1:
                line.append(symbol2)
            else:
                line.append(' ')
        line = '|' + '|'.join(line) + '|'
        print(pre + line)
    print(pre+"["+"|".join([str(i+1) for i in range(SEVEN)])+"]")


def initialize_utils():
    board = np.zeros((SEVEN, SIX))
    valid_moves(board)
    play(board, 0, 1)
    take_back(board, 0)
    n_in_a_board(board, 1)
    montecarlo(board, 1, 10)

# for one in [1, -1]:
#
#     diag = one*np.eye(N)
#     file = one*np.ones(N)
#
#     for i in range(SEVEN):
#         for j in range(SIX):
#
#             board = np.zeros((SEVEN, SIX))
#
#             if i + N < SEVEN and j + N < SIX:
#                 board[i:i+N, j:j+N] = diag
#                 assert _n_in_a_diag(board, one)
#                 assert n_in_a_board(board, one)
#
#             board = np.zeros((SEVEN, SIX))
#
#             if i + N <= SEVEN and j + N <= SIX:
#                 board[i:i+N, j:j+N] = np.flipud(diag)
#                 assert _n_in_a_diag(np.fliplr(board), one)
#                 assert n_in_a_board(board, one)
#
#             board = np.zeros((SEVEN, SIX))
#
#             if i + N <= SEVEN:
#                 board[i:i + N, j] = file
#                 assert _n_in_a_row(board, one)
#                 assert n_in_a_board(board, one)
#
#             board = np.zeros((SEVEN, SIX))
#
#             if j + N <= SIX:
#                 board[i, j:j+N] = file
#                 assert _n_in_a_col(board, one)
#                 assert n_in_a_board(board, one)
