# -*- coding:UTF-8 -*-
"""
 @Author: CNN
 @FileName: worker.py
 @DateTime: 2025-04-16 16:23
 @SoftWare: PyCharm
"""
# worker.py

from PyQt5.QtCore import QThread, pyqtSignal
from analysis_core import process_folder


class Worker(QThread):
    progress_signal = pyqtSignal(int)
    file_signal = pyqtSignal(str)
    finished = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, folder_path, input_fields, pre_data_list):
        super().__init__()
        self.folder_path = folder_path
        self.input_fields = input_fields
        self.pre_data_list = pre_data_list

    def run(self):
        error_occurred = False
        try:
            process_folder(
                folder_path=self.folder_path,
                input_fields=self.input_fields,
                pre_data_list=self.pre_data_list,
                progress_callback=self.progress_signal.emit,
                file_callback=self.file_signal.emit
            )
        except Exception as e:
            import traceback
            error_message = f'数据分析过程中出错：\n{type(e)}\n{e}'
            self.error_signal.emit(error_message)
            error_occurred = True  # 标记发生了异常
        finally:
            if not error_occurred:
                self.finished.emit()
