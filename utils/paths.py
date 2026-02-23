import os
import sys

def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = get_base_path()

PROTOTXT = os.path.join(
    BASE_DIR, "models", "face_detector", "deploy.prototxt"
)

MODEL = os.path.join(
    BASE_DIR, "models", "face_detector", "res10_300x300_ssd_iter_140000.caffemodel"
)

DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.pkl")