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
    QStackedWidget,
    QMessageBox,
    QListWidget,
)

from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QThread, Signal, Qt

from core.attendance import AttendanceManager
from utils.storage import load_embeddings, save_embeddings
from core.face_detector import FaceDetector
from core.embedder import FaceEmbedder
from core.enrollment import Enroller
from core.recognition import FaceRecognizer


# ---------------- CAMERA THREAD ---------------- #

class CameraThread(QThread):
    frame_signal = Signal(np.ndarray)
    enrollment_finished = Signal()

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
            if len(boxes) > 0:
                print(f"Debug: Faces detected: {len(boxes)}")
            else:
                pass # print("Debug: No faces detected")

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
                    
                    if not self.enroller.active:
                        self.enrollment_finished.emit()
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

        # Main Stacked Layout
        self.stack = QStackedWidget()
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.stack)
        self.setLayout(self.main_layout)

        # --- Page 1: Main Menu ---
        self.page_menu = QWidget()
        self.menu_layout = QVBoxLayout()
        self.page_menu.setLayout(self.menu_layout)

        self.btn_take_attendance = QPushButton("Take Attendance")
        self.btn_enroll = QPushButton("Enroll Student")
        self.btn_manage = QPushButton("Manage Students")
        
        # Styling for menu buttons
        self.btn_take_attendance.setMinimumHeight(50)
        self.btn_enroll.setMinimumHeight(50)
        self.btn_manage.setMinimumHeight(50)

        self.menu_layout.addStretch()
        self.menu_layout.addWidget(self.btn_take_attendance)
        self.menu_layout.addWidget(self.btn_enroll)
        self.menu_layout.addWidget(self.btn_manage)
        self.menu_layout.addStretch()

        # --- Page 2: Camera View ---
        self.page_camera = QWidget()
        self.camera_layout = QVBoxLayout()
        self.page_camera.setLayout(self.camera_layout)

        self.video_label = QLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignCenter) # Align center
        self.btn_stop = QPushButton("Stop")

        self.camera_layout.addWidget(self.video_label)
        self.camera_layout.addWidget(self.btn_stop)

        # --- Page 3: Manage Students ---
        self.page_manage = QWidget()
        self.manage_layout = QVBoxLayout()
        self.page_manage.setLayout(self.manage_layout)

        self.student_list = QListWidget()
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_back = QPushButton("Back to Menu")

        self.manage_layout.addWidget(QLabel("Enrolled Students:"))
        self.manage_layout.addWidget(self.student_list)
        self.manage_layout.addWidget(self.btn_delete)
        self.manage_layout.addWidget(self.btn_back)

        # Add pages to stack
        self.stack.addWidget(self.page_menu)
        self.stack.addWidget(self.page_camera)
        self.stack.addWidget(self.page_manage)

        # Camera Thread
        self.camera_thread = CameraThread()
        self.camera_thread.frame_signal.connect(self.update_image)
        self.camera_thread.enrollment_finished.connect(self.on_enrollment_finished)

        # Connections
        self.btn_take_attendance.clicked.connect(self.start_attendance)
        self.btn_enroll.clicked.connect(self.start_enrollment)
        self.btn_manage.clicked.connect(self.open_manage_page)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_delete.clicked.connect(self.delete_student)
        self.btn_back.clicked.connect(self.go_back_to_menu)

    def start_attendance(self):
        self.stack.setCurrentWidget(self.page_camera)
        self.btn_stop.setVisible(True)
        if not self.camera_thread.isRunning():
            self.camera_thread.start()

    def start_enrollment(self):
        name, ok = QInputDialog.getText(
            self,
            "Enroll Student",
            "Enter Student ID / Name:"
        )

        if ok and name.strip():
            self.camera_thread.enroller.start(name.strip())
            self.stack.setCurrentWidget(self.page_camera)
            self.btn_stop.setVisible(False)
            if not self.camera_thread.isRunning():
                self.camera_thread.start()

    def stop_camera(self):
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
        self.stack.setCurrentWidget(self.page_menu)
        self.video_label.setText("Camera Feed") # Reset label
        self.btn_stop.setVisible(True)

    def on_enrollment_finished(self):
        self.stop_camera()
        QMessageBox.information(self, "Enrollment", "Enrollment Complete!")
        # Update logic after enrollment if needed (e.g. if we were on manage page, but we aren't)

    def open_manage_page(self):
        self.stack.setCurrentWidget(self.page_manage)
        self.load_students()

    def load_students(self):
        self.student_list.clear()
        # Ensure we are using the latest DB state
        # (It is shared with camera_thread, so it should be up to date)
        if hasattr(self.camera_thread, 'embeddings_db'):
            for name in self.camera_thread.embeddings_db.keys():
                self.student_list.addItem(name)

    def delete_student(self):
        current_item = self.student_list.currentItem()
        if not current_item:
            return

        name = current_item.text()
        reply = QMessageBox.question(
            self, "Confirm Delete", f"Delete {name}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if name in self.camera_thread.embeddings_db:
                del self.camera_thread.embeddings_db[name]
                save_embeddings(self.camera_thread.embeddings_db)
                self.camera_thread.recognizer.update_db(self.camera_thread.embeddings_db)
                self.load_students()
                QMessageBox.information(self, "Deleted", f"Removed {name}")

    def go_back_to_menu(self):
        self.stack.setCurrentWidget(self.page_menu)

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
