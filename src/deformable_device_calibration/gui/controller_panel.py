# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import json

from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt6.QtWidgets import QSplitter, QWidget, QHBoxLayout, QVBoxLayout, QSpinBox, QDoubleSpinBox

from . import custom_widgets as cw


class ControlPanel(QWidget):
    Signal_set_laser = pyqtSignal(list, bool, float)
    Signal_video = pyqtSignal(bool, str)

    def __init__(self, config, logg, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.config = config
        self.logg = logg
        self._setup_ui()
        self.load_spinbox_values()
        self._set_signal_connections()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        self.device_panel = self._create_device_panel()
        self.acq_panel = self._create_acquisition_panel()

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.device_panel)
        splitter.addWidget(self.acq_panel)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _create_device_panel(self):
        group = cw.GroupWidget()

        cmos_scroll_area, cmos_scroll_layout = cw.create_scroll_area()
        laser_scroll_area, laser_scroll_layout = cw.create_scroll_area()

        self.QSpinBox_cmos_coordinate_x = cw.SpinBoxWidget(0, 2048, 4, 0)
        self.QSpinBox_cmos_coordinate_y = cw.SpinBoxWidget(0, 2048, 2, 0)
        self.QSpinBox_cmos_coordinate_nx = cw.SpinBoxWidget(0, 2048, 2, 2048)
        self.QSpinBox_cmos_coordinate_ny = cw.SpinBoxWidget(0, 2048, 4, 2048)
        self.QSpinBox_cmos_coordinate_bin = cw.SpinBoxWidget(0, 2048, 1, 1)
        self.QSpinBox_cmos_gain = cw.SpinBoxWidget(0, 300, 1, 0)
        self.QDoubleSpinBox_cmos_t_clean = cw.DoubleSpinBoxWidget(0, 10, 0.001, 4, 0.009)
        self.QDoubleSpinBox_cmos_t_exposure = cw.DoubleSpinBoxWidget(0, 10, 0.001, 4, 0.001)
        self.QDoubleSpinBox_cmos_t_standby = cw.DoubleSpinBoxWidget(0, 10, 0.001, 4, 0.050)

        cmos_scroll_layout.addRow(cw.LabelWidget(str('CMOS')))
        cmos_scroll_layout.addRow(cw.FrameWidget())
        cmos_scroll_layout.addRow(cw.LabelWidget(str('X')), self.QSpinBox_cmos_coordinate_x)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Y')), self.QSpinBox_cmos_coordinate_y)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Nx')), self.QSpinBox_cmos_coordinate_nx)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Ny')), self.QSpinBox_cmos_coordinate_ny)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Bin')), self.QSpinBox_cmos_coordinate_bin)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Gain')), self.QSpinBox_cmos_gain)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Clean / s')), self.QDoubleSpinBox_cmos_t_clean)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Exposure / ms')), self.QDoubleSpinBox_cmos_t_exposure)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Standby / s')), self.QDoubleSpinBox_cmos_t_standby)

        self.QRadioButton_laser_405 = cw.RadioButtonWidget('405 nm')
        self.QDoubleSpinBox_laser_power_405 = cw.DoubleSpinBoxWidget(0, 200, 0.1, 1, 0.0)
        self.QPushButton_laser_405 = cw.PushButtonWidget('ON', checkable=True)
        self.QRadioButton_laser_488 = cw.RadioButtonWidget('488 nm')
        self.QDoubleSpinBox_laser_power_488 = cw.DoubleSpinBoxWidget(0, 200, 0.1, 1, 0.0)
        self.QPushButton_laser_488 = cw.PushButtonWidget('ON', checkable=True)

        laser_scroll_layout.addRow(self.QRadioButton_laser_405, self.QDoubleSpinBox_laserpower_405)
        laser_scroll_layout.addRow(self.QPushButton_laser_405)
        laser_scroll_layout.addRow(self.QRadioButton_laser_488, self.QDoubleSpinBox_laser_power_488)
        laser_scroll_layout.addRow(self.QPushButton_laser_488)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(cmos_scroll_area)
        group_layout.addWidget(laser_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_acquisition_panel(self):
        group = cw.GroupWidget()
        acq_scroll_area, acq_scroll_layout = cw.create_scroll_area("G")

        self.QComboBox_imaging_detector_selection = cw.ComboBoxWidget(list_items=["CMOS"])
        self.QPushButton_video = cw.PushButtonWidget("Video", checkable=True)

        acq_scroll_layout.addWidget(cw.LabelWidget(str('Detector')), 0, 0, 1, 1)
        acq_scroll_layout.addWidget(self.QComboBox_imaging_detector_selection, 1, 0, 1, 1)
        acq_scroll_layout.addWidget(self.QPushButton_video, 1, 2, 1, 1)

        group_layout = QVBoxLayout(group)
        group_layout.addWidget(acq_scroll_area)
        group.setLayout(group_layout)
        return group

    def _set_signal_connections(self):
        self.QPushButton_laser_488.clicked.connect(self.set_laser_488)
        self.QPushButton_laser_405.clicked.connect(self.set_laser_405)
        self.QPushButton_video.clicked.connect(self.run_video)

    def get_cmos_roi(self):
        return [self.QSpinBox_cmos_coordinate_x.value(), self.QSpinBox_cmos_coordinate_y.value(),
                self.QSpinBox_cmos_coordinate_nx.value(), self.QSpinBox_cmos_coordinate_ny.value(),
                self.QSpinBox_cmos_coordinate_bin.value()]

    def get_cmos_gain(self):
        return self.QSpinBox_cmos_gain.value()

    def get_cmos_exposure(self):
        return self.QDoubleSpinBox_cmos_t_exposure.value()

    @pyqtSlot(bool)
    def set_laser_488(self, checked: bool):
        power = self.QDoubleSpinBox_laserpower_488.value()
        self.Signal_set_laser.emit(["488"], checked, power)

    @pyqtSlot(bool)
    def set_laser_405(self, checked: bool):
        power = self.QDoubleSpinBox_laserpower_405.value()
        self.Signal_set_laser.emit(["405"], checked, power)

    def get_lasers(self):
        lasers = []
        if self.QRadioButton_laser_405.isChecked():
            lasers.append(0)
        if self.QRadioButton_laser_488.isChecked():
            lasers.append(1)
        return lasers

    def get_cobolt_laser_power(self, laser):
        if laser == "405":
            return [self.QDoubleSpinBox_laserpower_405.value()]
        if laser == "488":
            return [self.QDoubleSpinBox_laserpower_488.value()]
        if "all" == laser:
            return [self.QDoubleSpinBox_laserpower_405.value(), self.QDoubleSpinBox_laserpower_488.value()]
        return None

    def get_detector(self):
        return self.QComboBox_imaging_detector_selection.currentIndex()

    @pyqtSlot()
    def run_video(self):
        vm = self.QComboBox_live_modes.currentText()
        if self.QPushButton_video.isChecked():
            self.Signal_video.emit(True, vm)
        else:
            self.Signal_video.emit(False, vm)

    def save_spinbox_values(self):
        values = {}
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, (QSpinBox, QDoubleSpinBox)):
                values[name] = obj.value()
        with open(self.config["ConWidget Path"], 'w') as f:
            json.dump(values, f, indent=4)

    def load_spinbox_values(self):
        try:
            with open(self.config["ConWidget Path"], 'r') as f:
                values = json.load(f)
            for name, value in values.items():
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.setValue(value)
        except FileNotFoundError:
            pass
