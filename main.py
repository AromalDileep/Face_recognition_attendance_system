import logging
import os
from ui.gui import run_gui

if __name__ == "__main__":
    # Create safe writable directory
    log_dir = os.path.join(os.getenv("LOCALAPPDATA"), "Attendance System")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "app.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    run_gui()