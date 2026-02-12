import cv2
import numpy as np

from core.attendance import AttendanceManager
from utils.storage import load_embeddings
from core.face_detector import FaceDetector
from core.embedder import FaceEmbedder
from core.enrollment import Enroller
from core.recognition import FaceRecognizer


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    detector = FaceDetector()
    embedder = FaceEmbedder()

    embeddings_db = load_embeddings()
    recognizer = FaceRecognizer(embeddings_db)

    # 🔹 Attendance manager
    attendance = AttendanceManager(cooldown_seconds=60)

    # 🔹 Enrollment with live recognizer update
    enroller = Enroller(
        embeddings_db,
        on_update=lambda db: recognizer.update_db(db)
    )

    print("✅ Press E to enroll | Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord("e") and not enroller.active:
            student_id = input("Enter Student ID / Name: ").strip()
            if student_id:
                enroller.start(student_id)

        elif key == ord("q"):
            break

        boxes = detector.detect(frame)

        for (x1, y1, x2, y2) in boxes:
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.resize(face, (160, 160))
            face = np.expand_dims(face, axis=0)

            embedding = embedder.get_embedding(face)

            # 🔴 Enrollment mode
            if enroller.active:
                enroller.process(embedding, frame)
                label = f"Enrolling: {enroller.name}"
                color = (0, 0, 255)

            # 🟢 Recognition + Attendance
            else:
                name, score = recognizer.recognize(embedding)

                if name != "Unknown":
                    attendance.mark_attendance(name)
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                label = f"{name} ({score:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        cv2.imshow("Face Attendance System", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
