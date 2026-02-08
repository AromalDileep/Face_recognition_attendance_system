import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROTOTXT = os.path.join(
    BASE_DIR, "models", "face_detector", "deploy.prototxt"
)

MODEL = os.path.join(
    BASE_DIR, "models", "face_detector", "res10_300x300_ssd_iter_140000.caffemodel"
)

DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.pkl")
