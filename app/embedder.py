import numpy as np
from keras_facenet import FaceNet


class FaceEmbedder:
    def __init__(self):
        # Load FaceNet model once
        self.embedder = FaceNet()

    def get_embedding(self, face):
        """
        face: numpy array of shape (1, 160, 160, 3)
        returns: embedding vector (512,)
        """
        embedding = self.embedder.embeddings(face)[0]
        return embedding
