import sys
import cv2
import numpy as np
import threading
from utils.serial_controller import trigger_motor

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
    QHBoxLayout,
    QListWidgetItem,
    QSplashScreen,
    QDialog,
    QGridLayout,
)

from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import QThread, Signal, Qt


# ---------------- CAMERA THREAD ---------------- #

# ---------------- MODEL LOADER THREAD ---------------- #

class ModelLoader(QThread):
    finished_loading = Signal(object) # param: dict of components

    def run(self):
        # Initialize heavy components here
        # Lazy imports to speed up splash screen appearance
        from core.face_detector import FaceDetector
        from core.embedder import FaceEmbedder
        from core.recognition import FaceRecognizer
        from core.attendance import AttendanceManager
        from utils.storage import load_embeddings

        components = {}
        
        print("Loading FaceDetector...")
        components['detector'] = FaceDetector()
        
        print("Loading FaceEmbedder...")
        components['embedder'] = FaceEmbedder()
        
        print("Loading Embeddings DB...")
        components['embeddings_db'] = load_embeddings()
        
        print("Initializing FaceRecognizer...")
        components['recognizer'] = FaceRecognizer(components['embeddings_db'])
        
        print("Initializing AttendanceManager...")
        components['attendance'] = AttendanceManager()
        
        self.finished_loading.emit(components)


# ---------------- CAMERA THREAD ---------------- #

class CameraThread(QThread):
    frame_signal = Signal(np.ndarray)
    enrollment_finished = Signal()

    def __init__(self, components):
        super().__init__()
        self.running = False

        self.detector = components['detector']
        self.embedder = components['embedder']
        self.embeddings_db = components['embeddings_db']
        self.recognizer = components['recognizer']
        self.attendance = components['attendance']

        self.recognizer = components['recognizer']
        self.attendance = components['attendance']

        from core.enrollment import Enroller
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

class PeriodSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Period")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.label = QLabel("Press 1-6 to select a period:")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Arial", 12, QFont.Bold))
        self.layout.addWidget(self.label)
        
        self.grid = QGridLayout()
        self.layout.addLayout(self.grid)
        
        self.selected_period = None
        
        positions = [(i, j) for i in range(2) for j in range(3)]
        
        for i, pos in zip(range(1, 7), positions):
            btn = QPushButton(f"Period {i}")
            btn.setMinimumHeight(40)
            btn.clicked.connect(lambda checked, p=i: self.select_period(p))
            self.grid.addWidget(btn, *pos)
            
    def keyPressEvent(self, event):
        key = event.key()
        if Qt.Key_1 <= key <= Qt.Key_6:
            self.select_period(key - 48) # Qt.Key_0 is 48
        else:
            super().keyPressEvent(event)
            
    def select_period(self, period_num):
        self.selected_period = f"Period-{period_num}"
        self.accept()


class MainWindow(QWidget):
    def __init__(self, components):
        super().__init__()
        self.components = components

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

        self.btn_take_attendance = QPushButton("&Take Attendance") # Alt+T
        self.btn_enroll = QPushButton("&Enroll Student") # Alt+E
        self.btn_manage = QPushButton("&Manage Students") # Alt+M
        
        # Styling for menu buttons
        self.btn_take_attendance.setMinimumHeight(50)
        self.btn_enroll.setMinimumHeight(50)
        self.btn_manage.setMinimumHeight(50)

        # Enable Enter key
        self.btn_take_attendance.setAutoDefault(True)
        self.btn_enroll.setAutoDefault(True)
        self.btn_manage.setAutoDefault(True)

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
        self.btn_stop = QPushButton("&Stop") # Alt+S
        self.btn_stop.setAutoDefault(True)

        self.camera_layout.addWidget(self.video_label)
        self.camera_layout.addWidget(self.btn_stop)

        # --- Page 3: Manage Students ---
        self.page_manage = QWidget()
        self.manage_layout = QVBoxLayout()
        self.page_manage.setLayout(self.manage_layout)

        self.student_list = QListWidget()
        self.btn_back = QPushButton("&Back to Menu") # Alt+B
        self.btn_back.setAutoDefault(True)

        self.manage_layout.addWidget(QLabel("Enrolled Students:"))
        self.manage_layout.addWidget(self.student_list)
        self.manage_layout.addWidget(self.btn_back)

        # Add pages to stack
        self.stack.addWidget(self.page_menu)
        self.stack.addWidget(self.page_camera)
        self.stack.addWidget(self.page_manage)

        # Camera Thread
        self.camera_thread = CameraThread(self.components)
        self.camera_thread.frame_signal.connect(self.update_image)
        self.camera_thread.enrollment_finished.connect(self.on_enrollment_finished)

        # Connections
        self.btn_take_attendance.clicked.connect(self.start_attendance)
        self.btn_enroll.clicked.connect(self.start_enrollment)
        self.btn_manage.clicked.connect(self.open_manage_page)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_back.clicked.connect(self.go_back_to_menu)

    def start_attendance(self):
        dialog = PeriodSelectionDialog(self)
        if dialog.exec():
            period = dialog.selected_period
            
            # Set the period in the backend
            if hasattr(self.camera_thread, 'attendance'):
                success = self.camera_thread.attendance.start_session(period)
                if not success:
                    QMessageBox.warning(self, "Error", f"Sheet '{period}' not found in Google Sheets!")
                    return

            # Trigger motor in background to not block UI
            threading.Thread(target=trigger_motor, daemon=True).start()

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
        if hasattr(self.camera_thread, 'embeddings_db'):
            for name in self.camera_thread.embeddings_db.keys():
                item = QListWidgetItem(self.student_list)
                widget = QWidget()
                layout = QHBoxLayout()
                layout.setContentsMargins(5, 5, 5, 5)
                
                label = QLabel(name)
                btn_delete = QPushButton("&Delete") # Alt+D
                btn_delete.setAutoDefault(True)
                btn_delete.setStyleSheet("background-color: #ffcccc; color: red;")
                # Use a default argument in lambda to capture the current name
                btn_delete.clicked.connect(lambda checked, n=name: self.delete_student(n))
                
                layout.addWidget(label)
                layout.addStretch()
                layout.addWidget(btn_delete)
                
                widget.setLayout(layout)
                item.setSizeHint(widget.sizeHint())
                self.student_list.setItemWidget(item, widget)

    def delete_student(self, name):
        reply = QMessageBox.question(
            self, "Confirm Delete", f"Delete {name}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            from utils.storage import save_embeddings
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

    # --- Loading Screen (Splash) ---
    pixmap = QPixmap(400, 200)
    pixmap.fill(Qt.black)
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    
    # Custom font for splash
    font = QFont()
    font.setPixelSize(14)
    font.setBold(True)
    splash.setFont(font)
    
    splash.showMessage(
        "Loading Face Recognition Models...\nThis may take a few seconds...", 
        Qt.AlignCenter, 
        Qt.white
    )
    splash.show()
    
    # Force event loop to render splash
    app.processEvents()

    # Define cleanup/start function
    # We must keep references to window to prevent garbage collection
    # Using a container list or global variable is a common trick in simple scripts,
    # or defining a class for the application controller.
    # Here we can use a closure.
    
    # Container for the main window to keep it alive
    refs = {}

    def start_app(components):
        window = MainWindow(components)
        window.show()
        refs['window'] = window # Keep reference
        splash.finish(window)

    # Start Loader
    loader = ModelLoader()
    loader.finished_loading.connect(start_app)
    loader.start()
    
    # Keep loader reference to prevent GC
    refs['loader'] = loader

    sys.exit(app.exec())
