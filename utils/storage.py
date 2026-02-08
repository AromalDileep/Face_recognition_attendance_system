import os
import pickle
from utils.paths import DATA_DIR, EMBEDDINGS_PATH

def load_embeddings():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_embeddings(data):
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(data, f)
