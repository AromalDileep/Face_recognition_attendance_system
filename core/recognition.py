import numpy as np
from numpy.linalg import norm


class FaceRecognizer:
    def __init__(self, embeddings_db, threshold=0.65):
        """
        embeddings_db: dict {name: [np.ndarray, ...]}
        threshold: cosine similarity threshold
        """
        self.db = embeddings_db
        self.threshold = threshold
        self.mean_db = self._build_mean_embeddings()

    def update_db(self, embeddings_db):
        self.db = embeddings_db
        self.mean_db = self._build_mean_embeddings()

    def _l2_normalize(self, v):
        return v / (norm(v) + 1e-10)

    def _build_mean_embeddings(self):
        """
        Compute one mean embedding per person
        """
        mean_db = {}

        for name, embeddings in self.db.items():
            if len(embeddings) == 0:
                continue

            embeddings = np.array(embeddings)
            embeddings = np.array([self._l2_normalize(e) for e in embeddings])

            mean_db[name] = np.mean(embeddings, axis=0)

        return mean_db

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b) + 1e-10)

    def recognize(self, embedding):
        """
        Returns: (name, similarity_score)
        """
        embedding = self._l2_normalize(embedding)

        best_name = "Unknown"
        best_score = -1.0

        for name, ref_emb in self.mean_db.items():
            score = self._cosine_similarity(embedding, ref_emb)

            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= self.threshold:
            return best_name, best_score

        return "Unknown", best_score
