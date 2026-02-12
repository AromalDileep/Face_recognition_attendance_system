import sys
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QInputDialog,
)

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QThread, Signal

from app.attendance import AttendanceManager
from utils.storage import load_embeddings
from app.face_detector import FaceDetector
from app.embedder import FaceEmbedder
from app.enrollment import Enroller
from app.recognition import FaceRecognizer


# ---------------- CAMERA THREAD ---------------- #

class CameraThread(QThread):
    frame_signal = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False

        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()

        self.embeddings_db = load_embeddings()
        self.recognizer = FaceRecognizer(self.embeddings_db)
        self.attendance = AttendanceManager()

        self.enroller = Enroller(
            self.embeddings_db,
            on_update=lambda db: self.recognizer.update_db(db),
        )

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return

        self.running = True

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            boxes = self.detector.detect(frame)

            for (x1, y1, x2, y2) in boxes:
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face = cv2.resize(face, (160, 160))
                face = np.expand_dims(face, axis=0)

                embedding = self.embedder.get_embedding(face)

                if self.enroller.active:
                    self.enroller.process(embedding, frame)
                    label = f"Enrolling: {self.enroller.name}"
                    color = (0, 0, 255)
                else:
                    name, score = self.recognizer.recognize(embedding)

                    if name != "Unknown":
                        self.attendance.mark_attendance(name)
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

            self.frame_signal.emit(frame)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ---------------- GUI WINDOW ---------------- #

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Recognition Attendance System")
        self.resize(900, 700)

        self.layout = QVBoxLayout()

        self.video_label = QLabel("Camera Feed")
        self.layout.addWidget(self.video_label)

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.enroll_button = QPushButton("Enroll Student")

        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.enroll_button)
        self.layout.addWidget(self.stop_button)

        self.setLayout(self.layout)

        self.camera_thread = CameraThread()

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.enroll_button.clicked.connect(self.start_enrollment)

        self.camera_thread.frame_signal.connect(self.update_image)

    def start_camera(self):
        if not self.camera_thread.isRunning():
            self.camera_thread.start()

    def stop_camera(self):
        if self.camera_thread.isRunning():
            self.camera_thread.stop()

    def start_enrollment(self):
        name, ok = QInputDialog.getText(
            self,
            "Enroll Student",
            "Enter Student ID / Name:"
        )

        if ok and name.strip():
            self.camera_thread.enroller.start(name.strip())


    def update_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_image = QImage(
            rgb_image.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        )

        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


# ---------------- ENTRY ---------------- #

def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
