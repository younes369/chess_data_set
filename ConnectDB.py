from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv
from bson import BSON


from sharedFunctions import findNextIdx, saveData, loading_status

load_dotenv()
uri = os.getenv("DB_URI")

# Create a new client and connect to the server
client = MongoClient(uri)
chess_data_set_DB = client["Chess_data_set"]
games_coll = chess_data_set_DB["Games"]

stats = chess_data_set_DB.command("dbstats")
total_size_in_MB = stats['dataSize'] / (1024 * 1024)
print("total DB size in MB: ", total_size_in_MB)


# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


current_size_in_MB = 0
document = 0

for document in games_coll.find().sort("_id"):
    moves = document.get('moves')
    positions = document.get('positions')
    document_size = len(BSON.encode(document))
    current_size_in_MB += document_size / (1024 * 1024)
    loading_status(current_size_in_MB, total_size_in_MB)
    saveData(moves, positions)


