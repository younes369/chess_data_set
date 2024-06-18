from stockfish import Stockfish
import chess
import random
from pprint import pprint
import numpy as np
import os
import glob
import time
from dotenv import load_dotenv
from helperFunctions import findNextIdx, saveData, checkEndCondition
load_dotenv()

stockfishPath = os.getenv("STOCK_FISH_PATH")
stockfish = Stockfish(path=stockfishPath)


def mineGames(numGames : int):
	"""mines numGames games of moves"""
	MAX_MOVES = 500 #don't continue games after this number

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
			currentGameMoves.append(move) #make sure to add str version of move before changing format
			move = chess.Move.from_uci(str(move)) #convert to format chess package likes
			board.push(move)
			stockfish.set_position(currentGameMoves)
			if (checkEndCondition(board)):
				print("game is over")
				break
		saveData(currentGameMoves, currentGamePositions)


mineGames(1)
