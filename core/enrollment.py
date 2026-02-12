import time
import cv2
from utils.storage import save_embeddings


class Enroller:
    def __init__(self, embeddings_db, max_samples=20, on_update=None):
        self.db = embeddings_db
        self.max_samples = max_samples
        self.on_update = on_update

        self.active = False
        self.name = None
        self.count = 0

    def start(self, name):
        self.name = name
        self.db.setdefault(name, [])
        self.count = 0
        self.active = True
        print(f"📸 Enrolling {name}...")

    def process(self, embedding, frame):
        if not self.active:
            return

        self.db[self.name].append(embedding)
        self.count += 1

        cv2.putText(
            frame,
            f"Enrolling {self.name}: {self.count}/{self.max_samples}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        time.sleep(0.15)

        if self.count >= self.max_samples:
            save_embeddings(self.db)
            self.active = False
            print(f"✅ Enrollment complete for {self.name}")

            # 🔥 THIS IS THE KEY LINE
            if self.on_update:
                self.on_update(self.db)
