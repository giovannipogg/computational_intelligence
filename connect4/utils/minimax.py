from typing import Callable, Tuple

import numpy as np
from numba import njit

from utils.connect import valid_moves, play, n_in_a_board, take_back, montecarlo

MAX = 3
SAMPLES = 20


@njit
def local_montecarlo(board, player, samples: int = SAMPLES):
    """Implementation of the montecarlo playout.


    """
    return -1, montecarlo(board, player, samples)


def minimax(board, player, look_ahead: int = MAX,
            estimator: Callable[[np.ndarray, int], Tuple[int, float]] = local_montecarlo) -> (int, float):
    """Returns the best move for player.

    `cap` is the number of turns it is going to evaluate:

        0 looks only implements the chained estimator (if given else plays at random).
        1 minimaxes its turn only (make winning move if any)
        2 minimaxes also other player turn (blocks opponent if possible)
        etc.
    """

    return _minimax(board, player, cap=look_ahead, estimator=estimator)


@njit
def _minimax(board, player, cap, estimator: Callable[[np.ndarray, int], Tuple[int, float]], depth=0,
             threshold: float = .01) -> (int, float):
    if depth >= cap:
        if estimator is None:
            return np.random.choice(valid_moves(board)), 0.
        return estimator(board, player)

    valid_moves_ = valid_moves(board)
    # if no valid moves:
    #   return tie
    if len(valid_moves_) == 0:
        return -1, 0.
    # Note that, if no valid moves, last move could not
    # have been winning (see below)

    # for all valid moves:
    #   if winning:
    #       return

    # print(f"Evauluating if winning ... (player={'X' if player == 1 else 'O'}, depth={depth})")

    for move in valid_moves_:
        play(board, move, player)
        winning = n_in_a_board(board, player)

        # print(move+1, winning)
        # print_board(board)
        # _ = input()

        take_back(board, move)
        if winning:
            return move, float(player)

    # if here, no right away winning move:
    # need recursion
    # for all valid moves:
    #   evaluate with recursion
    #   if evaluation == objective (better, is indistinguishable from):
    #       return objective (without recurring further)
    moves = []
    evaluations = []

    for move in valid_moves_:
        play(board, move, player)
        _, evaluation = _minimax(board, -player, cap, estimator, depth + 1)
        take_back(board, move)
        if np.abs(evaluation - player) < threshold:
            return move, float(player)
        evaluations.append(evaluation)
        moves.append(move)

    # print(f"Got these Evaluations ... (player={'X' if player == 1 else 'O'}, depth={depth})")
    # pprint({move:eval for move, eval in zip(moves, evaluations)})
    # print_board(board)
    # _ = input()

    evaluations = np.array(evaluations)
    moves = np.array(moves)

    amax = np.argmax(evaluations * player)
    if np.all(evaluations == 0) or np.all(evaluations == -player) or np.all(evaluations == player):
        amax = np.random.choice(len(moves))

    return moves[amax], evaluations[amax]


def initialize_minimax():
    board = np.zeros((7, 6))
    minimax(board, 1, look_ahead=1)
