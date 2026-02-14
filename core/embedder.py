import numpy as np
import onnxruntime as ort
import os


class FaceEmbedder:
    def __init__(self):
        # Load ONNX model
        # Assuming model is at models/facenet.onnx relative to project root
        # core/embedder.py -> ../models/facenet.onnx
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "facenet.onnx")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model found at {model_path}. Please run conversion script.")

        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def get_embedding(self, face):
        """
        face: numpy array of shape (1, 160, 160, 3) (RGB)
        returns: embedding vector (512,)
        """
        # Preprocessing: (x - 127.5) / 128.0
        face = face.astype(np.float32)
        face = (face - 127.5) / 128.0
        
        # Run inference
        embedding = self.session.run(None, {self.input_name: face})[0][0]
        return embedding
