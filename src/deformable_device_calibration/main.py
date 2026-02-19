# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import datetime
import json
import os
import sys

from PyQt6.QtWidgets import QApplication, QFileDialog

from . import executor, logger
from .computations import computator
from .devices import device
from .gui import main_window


def setup_folder():
    documents_dir = r'C:\\Users\\Public\\Data'
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    folder_name = f"{today_str}"
    full_path = os.path.join(documents_dir, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Created folder: {full_path}")
    else:
        print(f"Folder already exists: {full_path}")
    return full_path


def select_file_from_folder(parent, data_folder):
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "Select a File",
        data_folder,
        "All Files (*)"
    )
    return file_path if file_path else None


def load_config(dfd):
    with open(dfd, 'r') as f:
        cfg = json.load(f)
    return cfg


def change_config(values, dfd):
    with open(dfd, 'w') as f:
        json.dump(values, f)


class AppWrapper:
    def __init__(self):
        self.app = QApplication(sys.argv)  # Create an instance of QApplication
        self.app.setStyleSheet("""
        QWidget { background-color: #232629; color: #f0f0f0; font-size: 9pt; }
        QPushButton { background-color: #444; border: 1px solid #555; color: #f0f0f0; }
        QPushButton:hover { background-color: #666; }
        QLabel { color: #e0e0e0; }
        QSpinBox { background-color: #222; color: #f0f0f0; border: 1px solid #333; }
        """)
        self.data_folder = setup_folder()
        self.error_logger = logger.setup_logger(self.data_folder)

        selected_file = select_file_from_folder(None, self.data_folder)
        if not selected_file:
            self.error_logger.error("No file selected. Exiting.")
            sys.exit(0)
        self.error_logger.info(f"Selected file: {selected_file}")
        self.configs = load_config(selected_file)
        self.devices = device.DeviceManager(config=self.configs, logg=self.error_logger, path=self.data_folder,
                                            cf=selected_file)
        self.cmp = computator.ComputationManager(config=self.configs, logg=self.error_logger, path=self.data_folder)
        self.mwd = main_window.MainWindow(config=self.configs, logg=self.error_logger, path=self.data_folder)
        self.cmd_exc = executor.CommandExecutor(self.devices, self.mwd, self.cmp,
                                                self.data_folder, self.configs, self.error_logger, selected_file)
        self.mwd.aboutToClose.connect(self.close)

    def run(self):
        try:
            self.mwd.show()
            sys.exit(self.app.exec())
        except Exception as e:
            print(f"Fatal error: {e}")
            sys.exit(1)

    def close(self):
        self.devices.close()
        self.app.exit()


def main():
    app_wrapper = AppWrapper()
    app_wrapper.run()


if __name__ == '__main__':
    main()
