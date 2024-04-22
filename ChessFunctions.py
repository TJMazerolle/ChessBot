import numpy as np
import random
import chess

import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

def split_dims(board):
    board3d = np.zeros((14, 8, 8), dtype = np.int8)
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = np.unravel_index(square, (8,8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = np.unravel_index(square, (8,8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1
    aux = board.turn
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
        board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1
    board.turn = aux
    return board3d

def get_ai_move(board, depth, model):
    max_move = None
    max_eval = -np.inf
    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, -np.inf, np.inf, False, model)
        board.pop()
        if eval > max_eval:
            max_eval = eval
            max_move = move
    return max_move

def get_board_state(board, model):
    board3d = split_dims(board)
    board3d = np.expand_dims(board3d, 0)
    return model.predict(board3d, verbose=0)[0][0]

def get_best_moves(board, model):
    legal_moves_list = list(board.legal_moves)
    move_scores = np.zeros(len(legal_moves_list))
    for i, move in enumerate(board.legal_moves):
        new_board = board.copy()
        new_board.push(move)
        if new_board.is_checkmate():
            return [move]
        else:
            move_scores[i] = get_board_state(new_board, model)
    sorted_indices = np.argsort(move_scores)[::-1]
    sorted_moves = [legal_moves_list[i] for i in sorted_indices]
    return sorted_moves

#####################################################################################
def minimax_eval(board, model):
    board3d = split_dims(board)
    board3d = np.expand_dims(board3d, 0)
    return model.predict(board3d, verbose=0)[0][0]

def minimax(board, depth, alpha, beta, maximizing_player, model):
    if depth == 0 or board.is_game_over():
        return minimax_eval(board, model)
    if maximizing_player:
        max_eval = -np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, model)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval),
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, model)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval
#####################################################################################

def square_to_index(square):
    squares_index = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4,'f':5, 'g':6, 'h':7}
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]

def get_dataset(filename):
    data = np.load(filename)
    states, scores = data["states"], data["scores"]
    return states, scores

def build_model(conv_size, conv_depth):
    board3d = layers.Input(shape = (14,8,8))
    x = board3d
    for _ in range(conv_depth):
        x = layers.Conv2D(filters = conv_size, kernel_size = 3, 
                          padding = "same", activation = "relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, "relu")(x)
    x = layers.Dense(1, "sigmoid")(x)
    return models.Model(inputs = board3d, outputs = x)

def random_board(max_depth = 200):
    board = chess.Board()
    depth = random.randrange(0, max_depth)
    for _ in range(depth):
        all_moves = list(board.legal_moves)
        random_move = random.choice(all_moves)
        board.push(random_move)
        if board.is_game_over():
            break
    return board

def stockfish(board, depth):
    with chess.engine.SimpleEngine.popen_uci("./stockfish/stockfish-windows-x86-64-avx2") as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        score = result["score"].white().score()
        return score