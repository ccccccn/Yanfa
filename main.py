# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# main.py
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from ui_main import ExcelAnalysisApp

def resource_path(relative_path):
    """获取资源的绝对路径，适配打包后的路径"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def main():
    app = QApplication(sys.argv)
    icon_path = resource_path("resources/favicon.ico")
    window = ExcelAnalysisApp(icon_path)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

