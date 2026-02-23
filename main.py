import logging
from ui.gui import run_gui

if __name__ == "__main__":
    logging.basicConfig(
        filename='app.log', 
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    run_gui()
