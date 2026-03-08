import os
import sys
import shutil

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

APP_DATA_DIR = os.path.join(os.getenv("LOCALAPPDATA", os.path.expanduser("~")), "Attendance System")
DATA_DIR = os.path.join(APP_DATA_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

_OLD_DATA_DIR = os.path.join(BASE_DIR, "data")
_OLD_EMBEDDINGS = os.path.join(_OLD_DATA_DIR, "embeddings.pkl")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.pkl")

if not os.path.exists(EMBEDDINGS_PATH) and os.path.exists(_OLD_EMBEDDINGS):
    try:
        shutil.copy2(_OLD_EMBEDDINGS, EMBEDDINGS_PATH)
    except Exception:
        pass