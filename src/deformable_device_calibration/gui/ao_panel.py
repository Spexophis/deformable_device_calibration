# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import json

from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSpinBox, QDoubleSpinBox, QSplitter

from . import custom_widgets as cw


class AOPanel(QWidget):
    Signal_img_shwfs_base = pyqtSignal()
    Signal_dm_selection = pyqtSignal(str)
    Signal_push_actuator = pyqtSignal(int, float)
    Signal_set_zernike = pyqtSignal(str, int, float)
    Signal_set_dm = pyqtSignal(int)
    Signal_set_dm_flat = pyqtSignal()
    Signal_update_cmd = pyqtSignal()
    Signal_load_dm = pyqtSignal()
    Signal_save_dm = pyqtSignal()

    def __init__(self, config, logg, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.config = config
        self.logg = logg
        self._setup_ui()
        self.load_spinbox_values()
        self._set_signal_connections()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        self.parameter_panel = self._create_parameter_panel()
        self.shwfs_panel = self._create_wf_panel()
        self.dm_panel = self._create_dm_panel()

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.parameter_panel)
        splitter.addWidget(self.shwfs_panel)
        splitter.addWidget(self.dm_panel)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _create_parameter_panel(self):
        group = cw.GroupWidget()
        confocal_shwfs_parameters_scroll_area, confocal_shwfs_parameters_scroll_layout = cw.create_scroll_area()
        interferometry_wfs_parameters_scroll_area, interferometry_wfs_parameters_scroll_layout = cw.create_scroll_area()

        self.QLabel_wfrmd_foc = cw.LabelWidget(str('Method'))
        self.QComboBox_wfrmd_foc = cw.ComboBoxWidget(list_items=['correlation', 'iterative', 'gaussianfit'], length=100)
        self.QSpinBox_base_xcenter_foc = cw.SpinBoxWidget(0, 2048, 1, 1024)
        self.QSpinBox_base_ycenter_foc = cw.SpinBoxWidget(0, 2048, 1, 1024)
        self.QSpinBox_offset_xcenter_foc = cw.SpinBoxWidget(0, 2048, 1, 1024)
        self.QSpinBox_offset_ycenter_foc = cw.SpinBoxWidget(0, 2048, 1, 1024)
        self.QSpinBox_n_lenslets_x_foc = cw.SpinBoxWidget(0, 64, 1, 14)
        self.QSpinBox_n_lenslets_y_foc = cw.SpinBoxWidget(0, 64, 1, 14)
        self.QSpinBox_spacing_foc = cw.SpinBoxWidget(0, 64, 1, 26)
        self.QSpinBox_radius_foc = cw.SpinBoxWidget(0, 64, 1, 12)
        self.QDoubleSpinBox_foc_background = cw.DoubleSpinBoxWidget(0, 1, 0.01, 2, 0.1)
        self.QPushButton_img_shwfs_base = cw.PushButtonWidget('Set Base', enable=True)

        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('SHWFS')))
        confocal_shwfs_parameters_scroll_layout.addRow(cw.FrameWidget())
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Method')), self.QComboBox_wfrmd_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('X_center (Base)')),
                                                       self.QSpinBox_base_xcenter_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Y_center (Base)')),
                                                       self.QSpinBox_base_ycenter_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('X_center (Offset)')),
                                                       self.QSpinBox_offset_xcenter_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Y_center (Offset)')),
                                                       self.QSpinBox_offset_ycenter_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Lenslet X')),
                                                       self.QSpinBox_n_lenslets_x_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Lenslet Y')),
                                                       self.QSpinBox_n_lenslets_y_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Spacing')), self.QSpinBox_spacing_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Radius')), self.QSpinBox_radius_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Background')),
                                                       self.QDoubleSpinBox_foc_background)
        confocal_shwfs_parameters_scroll_layout.addRow(self.QPushButton_img_shwfs_base)

        self.QSpinBox_x_int_center = cw.SpinBoxWidget(0, 2048, 1, 1024)
        self.QSpinBox_y_int_center = cw.SpinBoxWidget(0, 2048, 1, 1024)
        self.QSpinBox_nxh_int = cw.SpinBoxWidget(0, 2048, 1, 128)
        self.QSpinBox_nyh_int = cw.SpinBoxWidget(0, 2048, 1, 128)
        self.QSpinBox_int_radius = cw.SpinBoxWidget(0, 2048, 1, 128)
        self.QRadioButton_mask = cw.RadioButtonWidget('Mask')
        self.QRadioButton_wrap = cw.RadioButtonWidget('Phase Wrapping')

        interferometry_wfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Interferometry')))
        interferometry_wfs_parameters_scroll_layout.addRow(cw.FrameWidget())
        interferometry_wfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('X_center')), self.QSpinBox_x_int_center)
        interferometry_wfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Y_center')), self.QSpinBox_y_int_center)
        interferometry_wfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('nX_half')), self.QSpinBox_nxh_int)
        interferometry_wfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('nY_half')), self.QSpinBox_nyh_int)
        interferometry_wfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Radius')), self.QSpinBox_int_radius)
        interferometry_wfs_parameters_scroll_layout.addRow(self.QRadioButton_mask)
        interferometry_wfs_parameters_scroll_layout.addRow(self.QRadioButton_wrap)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(confocal_shwfs_parameters_scroll_area)
        group_layout.addWidget(interferometry_wfs_parameters_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_wf_panel(self):
        group = cw.GroupWidget()
        image_wfr_scroll_area, image_wfr_scroll_layout = cw.create_scroll_area()

        self.lcdNumber_wfmax_img = cw.LCDNumberWidget()
        self.lcdNumber_wfmin_img = cw.LCDNumberWidget()
        self.lcdNumber_wfrms_img = cw.LCDNumberWidget()

        image_wfr_scroll_layout.addRow(cw.LabelWidget(str('Wavefront MAX')), self.lcdNumber_wfmax_img)
        image_wfr_scroll_layout.addRow(cw.LabelWidget(str('Wavefront MIN')), self.lcdNumber_wfmin_img)
        image_wfr_scroll_layout.addRow(cw.LabelWidget(str('Wavefront RMS')), self.lcdNumber_wfrms_img)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(image_wfr_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_dm_panel(self):
        group = cw.GroupWidget()
        dm_scroll_area, dm_scroll_layout = cw.create_scroll_area("G")

        self.QComboBox_dms = cw.ComboBoxWidget(list_items=[])
        self.QComboBox_wfsmd = cw.ComboBoxWidget(list_items=['modal', 'phase', 'zonal'], length=64)
        self.QSpinBox_actuator = cw.SpinBoxWidget(0, 96, 1, 0)
        self.QDoubleSpinBox_actuator_push = cw.DoubleSpinBoxWidget(-1, 1, 0.005, 3, 0)
        self.QPushButton_push_actuator = cw.PushButtonWidget('Push')
        self.QSpinBox_zernike_mode = cw.SpinBoxWidget(0, 100, 1, 0)
        self.QDoubleSpinBox_zernike_mode_amp = cw.DoubleSpinBoxWidget(-10, 10, 0.002, 3, 0)
        self.QPushButton_set_zernike_mode = cw.PushButtonWidget('Set Zernike')
        self.QComboBox_cmd = cw.ComboBoxWidget(list_items=[])
        self.QPushButton_setDM = cw.PushButtonWidget('Set DM')
        self.QPushButton_load_dm = cw.PushButtonWidget('Load DM')
        self.QPushButton_update_cmd = cw.PushButtonWidget('Add DM')
        self.QPushButton_save_dm = cw.PushButtonWidget('Save DM')
        self.QPushButton_change_dm_flat = cw.PushButtonWidget('Save Flat')

        dm_scroll_layout.addWidget(cw.LabelWidget(str('DM')), 0, 0, 1, 1)
        dm_scroll_layout.addWidget(self.QComboBox_dms, 0, 1, 1, 1)
        dm_scroll_layout.addWidget(cw.LabelWidget(str('Method')), 0, 2, 1, 1)
        dm_scroll_layout.addWidget(self.QComboBox_wfsmd, 0, 3, 1, 1)
        dm_scroll_layout.addWidget(cw.LabelWidget(str('Actuator')), 1, 0, 1, 1)
        dm_scroll_layout.addWidget(self.QSpinBox_actuator, 1, 1, 1, 1)
        dm_scroll_layout.addWidget(cw.LabelWidget(str('Push')), 2, 0, 1, 1)
        dm_scroll_layout.addWidget(self.QDoubleSpinBox_actuator_push, 2, 1, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_push_actuator, 3, 0, 1, 1)
        dm_scroll_layout.addWidget(cw.LabelWidget(str('Zernike Mode')), 1, 2, 1, 1)
        dm_scroll_layout.addWidget(self.QSpinBox_zernike_mode, 1, 3, 1, 1)
        dm_scroll_layout.addWidget(cw.LabelWidget(str('Amplitude')), 2, 2, 1, 1)
        dm_scroll_layout.addWidget(self.QDoubleSpinBox_zernike_mode_amp, 2, 3, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_set_zernike_mode, 3, 2, 1, 1)
        dm_scroll_layout.addWidget(self.QComboBox_cmd, 4, 0, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_setDM, 4, 1, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_load_dm, 3, 3, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_update_cmd, 4, 2, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_change_dm_flat, 4, 3, 1, 1)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(dm_scroll_area)
        group.setLayout(group_layout)
        return group

    def _set_signal_connections(self):
        self.QPushButton_img_shwfs_base.clicked.connect(self.img_wfs_base)
        self.QComboBox_dms.currentIndexChanged.connect(self.select_dm)
        self.QPushButton_push_actuator.clicked.connect(self.push_dm_actuator)
        self.QPushButton_set_zernike_mode.clicked.connect(self.set_dm_zernike)
        self.QPushButton_setDM.clicked.connect(self.set_dm_acts)
        self.QPushButton_update_cmd.clicked.connect(self.update_dm_cmd)
        self.QPushButton_load_dm.clicked.connect(self.load_dm_file)
        self.QPushButton_save_dm.clicked.connect(self.save_dm_cmd)
        self.QPushButton_change_dm_flat.clicked.connect(self.change_dm_flat)

    def display_img_wf_properties(self, properties):
        self.lcdNumber_wfmin_img.display(properties[0])
        self.lcdNumber_wfmax_img.display(properties[1])
        self.lcdNumber_wfrms_img.display(properties[2])

    def get_parameters_int(self):
        return (self.QSpinBox_x_int_center.value(), self.QSpinBox_y_int_center.value(),
                self.QSpinBox_nxh_int.value(), self.QSpinBox_nyh_int.value(), self.QSpinBox_int_radius.value(),
                self.QRadioButton_mask.isChecked(), self.QRadioButton_wrap.isChecked())

    def get_parameters_foc(self):
        return (self.QSpinBox_base_xcenter_foc.value(), self.QSpinBox_base_ycenter_foc.value(),
                self.QSpinBox_offset_xcenter_foc.value(), self.QSpinBox_offset_ycenter_foc.value(),
                self.QSpinBox_n_lenslets_x_foc.value(), self.QSpinBox_n_lenslets_y_foc.value(),
                self.QSpinBox_spacing_foc.value(), self.QSpinBox_radius_foc.value(),
                self.QDoubleSpinBox_foc_background.value())

    def get_gradient_method_img(self):
        return self.QComboBox_wfrmd_foc.currentText()

    def get_img_wfs_method(self):
        return self.QComboBox_wfsmd.currentText()

    @pyqtSlot()
    def img_wfs_base(self):
        self.Signal_img_shwfs_base.emit()

    @pyqtSlot()
    def select_dm(self):
        dn = self.QComboBox_dms.currentText()
        self.Signal_dm_selection.emit(dn)

    @pyqtSlot()
    def push_dm_actuator(self):
        n = self.QSpinBox_actuator.value()
        a = self.QDoubleSpinBox_actuator_push.value()
        self.Signal_push_actuator.emit(n, a)

    @pyqtSlot()
    def set_dm_zernike(self):
        md = self.get_img_wfs_method()
        ind, amp = self.get_zernike_mode()
        self.Signal_set_zernike.emit(md, ind, amp)

    @pyqtSlot()
    def set_dm_acts(self):
        i = self.get_cmd_index()
        self.Signal_set_dm.emit(i)

    @pyqtSlot()
    def update_dm_cmd(self):
        self.Signal_update_cmd.emit()

    @pyqtSlot()
    def change_dm_flat(self):
        self.Signal_set_dm_flat.emit()

    @pyqtSlot()
    def load_dm_file(self):
        self.Signal_load_dm.emit()

    @pyqtSlot()
    def save_dm_cmd(self):
        self.Signal_save_dm.emit()

    def get_actuator(self):
        return self.QSpinBox_actuator.value(), self.QDoubleSpinBox_actuator_push.value()

    def get_zernike_mode(self):
        return self.QSpinBox_zernike_mode.value(), self.QDoubleSpinBox_zernike_mode_amp.value()

    def get_dm_selection(self):
        return self.QComboBox_dms.currentText()

    def get_cmd_index(self):
        return self.QComboBox_cmd.currentIndex()

    def update_cmd_index(self, wst=True):
        item = '{}'.format(self.QComboBox_cmd.count())
        self.QComboBox_cmd.addItem(item)
        if wst:
            self.QComboBox_cmd.setCurrentIndex(self.QComboBox_cmd.count() - 1)

    def save_spinbox_values(self):
        values = {}
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, (QSpinBox, QDoubleSpinBox)):
                values[name] = obj.value()
        with open(self.config["AOWidget Path"], 'w') as f:
            json.dump(values, f, indent=4)

    def load_spinbox_values(self):
        try:
            with open(self.config["AOWidget Path"], 'r') as f:
                values = json.load(f)
            for name, value in values.items():
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.setValue(value)
        except FileNotFoundError:
            pass
