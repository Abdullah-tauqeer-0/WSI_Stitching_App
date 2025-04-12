import sys
from PyQt5.QtWidgets import QApplication
from gui.ui import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

# Refactor pending for v2
# Fixed edge case 161