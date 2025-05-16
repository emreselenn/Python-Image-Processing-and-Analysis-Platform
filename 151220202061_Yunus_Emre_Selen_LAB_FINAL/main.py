import sys
from PyQt5.QtWidgets import QApplication
from ui import MainWindowController

def main():
    """
    @brief Initializes the application and shows the main window.

    Starts the application, creates the main window, and enters the event loop.
    """
    app = QApplication(sys.argv)
    window = MainWindowController()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
