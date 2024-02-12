import numpy as np
import chess



def runGame(numMoves, filename = "movesAndPositions1.npy"):
 """run a game you stored"""
 testing = np.load(f"data/{filename}")
 moves = testing[:, 0]
 if (numMoves > len(moves)):
  print("Must enter a lower number of moves than maximum game length. Game length here is: ", len(moves))
  return

 testBoard = chess.Board()

 for i in range(numMoves):
  move = moves[i]
  testBoard.push_san(move)
 return testBoard

Board = runGame(80, "movesAndPositions3.npy")
print(Board) 

