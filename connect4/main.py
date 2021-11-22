from utils.connect import SEVEN, SIX, print_board, play, initialize_utils, n_in_a_board, valid_moves
from utils.screens import loading_screen, welcome_screen, main_menu_screen, thanks_screen, \
    won_screen, tie_screen, lost_screen
from os import system, name
from utils.minimax import minimax, initialize_minimax

import numpy as np


def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')
        
        
def input_buffer(exitable=True):
    
    print()
    if exitable:
        print("                        [E]xit")
    else:
        print("press ENTER to continue...")
    print()
    print()


def play_loop(human_player=1) -> int:
    board = np.zeros((SEVEN, SIX))
    player = 1
    valid_inputs = [str(i + 1) for i in range(SEVEN)] + ['E']

    hint = ">> "
    in_progress, result = True, 0

    clear()
    print_board(board)
    input_buffer()

    while in_progress:

        valid_moves_ = valid_moves(board)
        # move, eval = minimax(board, player)
        # print(f"Evaluation: {eval:.1f}\tMove: {move+1}\n")

        if player == human_player:
            last_input = input(hint).upper()

            if last_input == 'E':
                clear()
                print(thanks_screen)
                exit()
            # if last_input == 'B':
            #     break

            try:
                last_input = int(last_input) - 1

                if last_input in valid_moves_:
                    play(board, last_input, human_player)
                    hint = ">> "
                    player = -player
                else:
                    last_input += 1
                    raise ValueError

            except ValueError:
                if str(last_input) not in valid_inputs:
                    hint = f"Invalid input {last_input}: valid inputs are {[a for a in valid_inputs]}\n>> "
                else:
                    hint = f"Column {last_input} is full!\n>> "
                continue
        else:
            move, eval_ = minimax(board, -human_player)
            play(board, move, -human_player)
            player = -player

        if n_in_a_board(board, human_player):
            in_progress, result = False, human_player
        elif n_in_a_board(board, -human_player):
            in_progress, result = False, -human_player
        elif len(valid_moves(board)) == 0:
            in_progress, result = False, 0

        clear()
        print_board(board)
        if not in_progress:
            input_buffer(exitable=False)
            _ = input(hint)
        else:
            input_buffer()

    return result


def replay_loop():
    human_player = 1
    result = play_loop(human_player)

    hint = ">> "
    while True:
        clear()
        if result == human_player:
            print(won_screen)
        elif result == -human_player:
            print(lost_screen)
        else:
            print(tie_screen)
        last_input = input(hint).upper()
        if last_input == 'E':
            clear()
            print(thanks_screen)
            exit()
        elif last_input == 'P':
            human_player = -human_player
            result = play_loop(human_player)
        else:
            hint = f"Invalid input {last_input}: valid inputs are {['P', 'E']}\n>> "


def main_loop():
    actions = {
        'P': replay_loop
    }

    hint = ">> "

    clear()
    print(loading_screen)

    initialize_utils()
    initialize_minimax()

    clear()
    print(welcome_screen)
    _ = input()

    while True:
        clear()
        print(main_menu_screen)
        last_input = input(hint).upper()
        if last_input == 'E':
            clear()
            print(thanks_screen)
            exit()
        if last_input in actions:
            actions[last_input]()
        else:
            hint = f"Invalid input {last_input}: valid inputs are {list(actions.keys()) + ['E']}\n>> "


if __name__ == "__main__":
    main_loop()
