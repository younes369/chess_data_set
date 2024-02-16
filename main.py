from stockfish import Stockfish
import chess
import random
from pprint import pprint
import numpy as np
import os
import glob
import time


from multiprocessing import Pool


stockfish = Stockfish(path=r"/home/youneszied/programming/chess_data_set/stockfish-ubuntu-x86-64-modern/stockfish/stockfish-ubuntu-x86-64-modern")


#helper functions:
def checkEndCondition(board):
 if (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition() or board.can_claim_fifty_moves() or board.can_claim_draw()):
  return True
 return False

#save
def findNextIdx():
 files = (glob.glob(r"/home/youneszied/programming/chess_data_set/data/*.npy"))
 if (len(files) == 0):
  return 1 #if no files, return 1
 highestIdx = 0
 for f in files:
  file = f
  currIdx = file.split("movesAndPositions")[-1].split(".npy")[0]
  highestIdx = max(highestIdx, int(currIdx))

 return int(highestIdx)+1

def saveData(moves, positions):
 moves = np.array(moves).reshape(-1, 1)
 positions = np.array(positions).reshape(-1,1)
 movesAndPositions = np.concatenate((moves, positions), axis = 1)
 nextIdx = findNextIdx()
 np.save(f"data/RawData/movesAndPositions{nextIdx}.npy", movesAndPositions)
 print("Saved successfully")


def mineGames(numGames : int,MAX_MOVES = 500):
 """mines numGames games of moves"""

 for i in range(numGames):
  currentGameMoves = []
  currentGamePositions = []
  board = chess.Board()
  stockfish.set_position([])

  for i in range(MAX_MOVES):
   #randomly choose from those 3 moves
   moves = stockfish.get_top_moves(3)
   #if less than 3 moves available, choose first one, if none available, exit
   if (len(moves) == 0):
    print("game is over")
    break
   elif (len(moves) == 1):
    move = moves[0]["Move"]
   elif (len(moves) == 2):
    move = random.choices(moves, weights=(80, 20), k=1)[0]["Move"]
   else:
    move = random.choices(moves, weights=(80, 15, 5), k=1)[0]["Move"]

   currentGamePositions.append(stockfish.get_fen_position())
   board.push_san(move)
   currentGameMoves.append(move)
   stockfish.set_position(currentGameMoves)
   if (checkEndCondition(board)):
    print("game is over")
    break
  saveData(currentGameMoves, currentGamePositions)



mineGames(1)