import numpy as np
import glob
import os
from dotenv import load_dotenv
import time
import sys


load_dotenv()
RawDataPath = os.getenv("RAW_DATA_PATH")

#helper functions:
def checkEndCondition(board):
 if (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition() or board.can_claim_fifty_moves() or board.can_claim_draw()):
  return True
 return False

#save
def findNextIdx():
 files = (glob.glob(f"{RawDataPath}/*.npy"))
 if (len(files) == 0):
  return 1 #if no files, return 1
 highestIdx = 0
 for f in files:
  file = f
  currIdx = file.split("movesAndPositions")[-1].split(".npy")[0]
  highestIdx = max(highestIdx, int(currIdx))
  
 print(highestIdx)
 return int(highestIdx)+1

def saveData(moves, positions):
 moves = np.array(moves).reshape(-1, 1)
 positions = np.array(positions).reshape(-1,1)
 movesAndPositions = np.concatenate((moves, positions), axis = 1)
 nextIdx = findNextIdx()
 np.save(f"{RawDataPath}/movesAndPositions{nextIdx}.npy", movesAndPositions)
#  print("Saved successfully")

# loading status function
def loading_status(current,total):
    percent_complete = (current / total) * 100
    sys.stdout.write(f'\rLoading... {percent_complete:.2f}%')
    sys.stdout.flush()
