# -*- coding:UTF-8 -*-
"""
 @Author: CNN
 @FileName: ui_main.py
 @DateTime: 2025-04-16 16:22
 @SoftWare: PyCharm
"""

# ui_main.py
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QLabel, QLineEdit,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
    QRadioButton, QProgressDialog, QApplication
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from worker import Worker
from config import LABELS


class ExcelAnalysisApp(QMainWindow):
    def __init__(self, icon_path=None):
        super().__init__()
        self.setWindowTitle('一次调频数据分析软件_V2.4.0')
        self.setGeometry(100, 100, 500, 600)
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 单选框区域
        radio_group = QHBoxLayout()
        radio_group.addWidget(QLabel("处理方式:"))

        self.radio_batch = QRadioButton("批量处理")
        self.radio_single = QRadioButton("单文件处理")
        self.radio_single.setChecked(True)

        radio_group.addWidget(self.radio_single)
        radio_group.addWidget(self.radio_batch)

        # 单选框事件连接
        self.radio_batch.toggled.connect(lambda: self.DealFileType(True))
        self.radio_single.toggled.connect(lambda: self.DealFileType(False))

        h_box = QHBoxLayout()
        h_box.addLayout(radio_group)
        layout.addLayout(h_box)

        # 导入按钮
        self.import_button = QPushButton('导入单日csv文件')
        self.import_button.clicked.connect(self.importCSV)
        layout.addWidget(self.import_button)

        self.import_button1 = QPushButton('导入文件夹')
        self.import_button1.clicked.connect(self.importExcel)
        layout.addWidget(self.import_button1)

        # 初始状态（默认是单文件）
        self.import_button1.setVisible(False)

        self.label1 = QLabel('等待导入预测调频曲线数据...')
        layout.addWidget(self.label1)

        import_data_button = QPushButton('导入数据文件')
        import_data_button.clicked.connect(self.importPreData)
        layout.addWidget(import_data_button)

        self.input_fields = {}
        for label in LABELS:
            if label == '采样周期（ms）':
                h_layout = QHBoxLayout()
                radio_group = QHBoxLayout()
                radio_group.addWidget(QLabel("是否定间隔:"))
                self.radio_yes = QRadioButton("是")
                self.radio_no = QRadioButton("否")
                radio_group.addWidget(self.radio_yes)
                radio_group.addWidget(self.radio_no)

                input_group = QHBoxLayout()
                input_group.addWidget(QLabel(label))
                lineEdit = QLineEdit()
                input_group.addWidget(lineEdit)
                self.input_fields[label] = lineEdit
                self.sample_cycle_edit = lineEdit

                self.radio_yes.setChecked(True)
                self.radio_yes.toggled.connect(lambda: self.toggleSampleCycle(True))
                self.radio_no.toggled.connect(lambda: self.toggleSampleCycle(False))

                h_layout.addLayout(radio_group, 1)
                h_layout.addLayout(input_group, 7)
                layout.addLayout(h_layout)
            else:
                layout.addWidget(QLabel(label))
                lineEdit = QLineEdit()
                layout.addWidget(lineEdit)
                self.input_fields[label] = lineEdit

        start_button = QPushButton('开始处理数据')
        start_button.clicked.connect(self.startExecution)
        layout.addWidget(start_button)

        self.current_file_label = QLabel('当前处理文件：无')
        layout.addWidget(self.current_file_label)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def DealFileType(self, is_batch):
        self.import_button.setVisible(not is_batch)
        self.import_button1.setVisible(is_batch)

    def toggleSampleCycle(self, enabled):
        self.sample_cycle_edit.setEnabled(enabled)
        self.sample_cycle_edit.setStyleSheet(
            "" if enabled else "QLineEdit:disabled { background: #f5f5f5; color: #a0a0a0; border: 1px solid #d0d0d0; }"
        )

    def importCSV(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择csv文件', '', 'CSV Files (*.csv);;All Files ()')
        if file_path:
            self.folder_path = file_path
            self.label.setText(f'已导入文件：{file_path}')

    def importExcel(self):
        folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹')
        if folder_path:
            self.folder_path = folder_path
            self.label.setText(f'已导入文件夹：{folder_path}')

    def importPreData(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择预测数据文件', '', 'Excel Files (*.xlsx *.xls)')
        if file_path:
            self.data_excel_file = file_path
            self.label1.setText(f'已导入数据文件：{file_path}')
            try:
                self.progress_dialog = QProgressDialog("正在处理预测数据...", None, 0, 0, self)
                self.progress_dialog.setWindowModality(Qt.ApplicationModal)
                self.progress_dialog.setWindowTitle("请稍候")
                self.progress_dialog.setCancelButton(None)
                self.progress_dialog.setMinimumDuration(0)
                self.progress_dialog.show()
                QApplication.processEvents()

                import pandas as pd
                pre_data_pd = pd.read_excel(file_path)
                self.pre_data_list = [row[0] for row in pre_data_pd.values.tolist()]

                self.progress_dialog.close()
                QMessageBox.information(self, '完成', f'预测曲线数据处理完成，已加载 {len(self.pre_data_list)} 条数据')
            except Exception as e:
                self.progress_dialog.close()
                QMessageBox.critical(self, '错误', f'读取数据出错：{e}')

    def startExecution(self):
        if not hasattr(self, 'folder_path'):
            QMessageBox.warning(self, '警告', '请导入分析文件夹')
            return
        for label, widget in self.input_fields.items():
            if widget == self.sample_cycle_edit and not widget.isEnabled():
                continue
            if not widget.text().strip():
                QMessageBox.warning(self, '警告', f'【{label}】不能为空')
                return

        self.worker = Worker(self.folder_path, self.input_fields, self.pre_data_list)
        self.worker.progress_signal.connect(self.updateProgress)
        self.worker.file_signal.connect(self.updateFileLabel)
        self.worker.finished.connect(self.analysisFinished)
        self.worker.error_signal.connect(self.handlerWorkerError)
        self.worker.start()

    def handlerWorkerError(self, error_message):
        QMessageBox.critical(self, '错误', error_message)
        try:
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait()
        except Exception as e:
            print("终止线程异常：", e)
        self.current_file_label.setText('当前处理文件：无')

    def updateProgress(self, value):
        pass  # 可加进度条实现

    def updateFileLabel(self, text):
        self.current_file_label.setText(text)

    def analysisFinished(self):
        try:
            QMessageBox.information(self, '完成', '数据分析完成')
            self.current_file_label.setText('当前处理文件：无')
        except Exception as e:
            print("分析完成处理异常：", e)
