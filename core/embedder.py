import numpy as np
import onnxruntime as ort
import os
import sys
import logging


class FaceEmbedder:
    def __init__(self):
        """
        Loads FaceNet ONNX model.
        Works in both normal Python and PyInstaller frozen EXE.
        """

        # Detect if running inside PyInstaller
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        model_path = os.path.join(base_dir, "models", "facenet.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model NOT found at: {model_path}"
            )

        logging.info(f"Loading ONNX model from: {model_path}")

        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, face):
        """
        face: numpy array of shape (1, 160, 160, 3) (RGB)
        returns: embedding vector (512,)
        """

        # Normalize input
        face = face.astype(np.float32)
        face = (face - 127.5) / 128.0

        # Run inference
        embedding = self.session.run(None, {self.input_name: face})[0][0]

        return embedding