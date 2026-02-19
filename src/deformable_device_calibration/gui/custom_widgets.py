# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from PyQt6 import QtWidgets, QtGui, QtCore


class ToolBarWidget(QtWidgets.QToolBar):
    def __init__(self):
        super().__init__()
        self.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Weight.Bold))
        self.setStyleSheet('QToolBar {background-color: #121212; color: white;}')


class DockWidget(QtWidgets.QDockWidget):
    def __init__(self, name=''):
        super().__init__(name)
        self.setStyleSheet('''
            QDockWidget {
                background-color: #121212;
                font-weight: bold;
                font-size: 9pt;
                color: #CCCCCC;
            }
            QDockWidget::title {
                text-align: center;
                background-color: #1E1E1E;
                padding: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QDockWidget::close-button {
                background-color: #666666;
                icon-size: 12px;
            }
            QDockWidget::close-button:hover {
                background-color: #ff5555;
            }
        ''')
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)


class GroupWidget(QtWidgets.QGroupBox):
    def __init__(self, name=''):
        super().__init__(name)
        self.setStyleSheet('''
            QGroupBox {
                background-color: #1E1E1E;
                border: 0px solid #1E1E1E;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
                margin-top: 0ex;
                font-weight: bold;
                font-size: 9pt;
                color: #CCCCCC;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 0px;
            }
        ''')
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)


class FileDialogWidget(QtWidgets.QFileDialog):
    def __init__(self, name="Save File", file_filter="All Files (*)", default_dir=""):
        super().__init__()
        options = QtWidgets.QFileDialog.Option.DontUseNativeDialog
        self.setOptions(options)
        if name == "Save File":
            self.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        elif name == "Open File":
            self.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        self.setNameFilters([file_filter])
        self.setWindowTitle(name)
        self.setDirectory(default_dir)
        self.setStyleSheet("""
            QFileDialog {
                background-color: #121212;
                color: white;
            }
            QFileDialog QLabel {
                color: white;
            }
            QFileDialog QLineEdit {
                background-color: #1E1E1E;
                color: white;
                selection-background-color: #0096FF;
            }
            QFileDialog QPushButton {
                background-color: #1E1E1E;
                color: white;
                padding: 5px;
                min-width: 80px;
            }
            QFileDialog QPushButton:hover {
                background-color: #0096FF;
            }
        """)


class ScrollAreaWidget(QtWidgets.QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        self.setStyleSheet("QScrollArea {background-color: #1E1E1E; color: white;}")


class FrameWidget(QtWidgets.QFrame):
    def __init__(self, h=True):
        super().__init__()
        if h:
            self.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        else:
            self.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)


class LabelWidget(QtWidgets.QLabel):
    def __init__(self, name=''):
        super().__init__(name)
        self.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Weight.Bold))
        self.setStyleSheet('background-color: #232629; color: #ECECEC; padding: 2px; border-radius: 2px;')
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)


class LCDNumberWidget(QtWidgets.QLCDNumber):
    def __init__(self, num=None, n=None):
        super().__init__()
        self.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Weight.Bold))
        self.setStyleSheet("""
            QLCDNumber {
                background-color: #121212;
                color: white;
                border: 1px solid #333333;
            }
        """)
        self.setSegmentStyle(QtWidgets.QLCDNumber.SegmentStyle.Flat)
        self.setDecMode()
        if num is not None:
            self.display(num)
        if n is not None:
            self.setDigitCount(n)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(self.sizeHint().width())
        self.setMaximumWidth(80)
        self.setMaximumHeight(32)


class SpinBoxWidget(QtWidgets.QSpinBox):

    def __init__(self, range_min, range_max, step, value):
        super().__init__()

        self.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Weight.Bold))
        self.setMinimumWidth(self.sizeHint().width())
        self.setMinimumHeight(self.sizeHint().height())
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        max_value_width = self.fontMetrics().horizontalAdvance(str(self.maximum()))
        self.setMaximumWidth(max_value_width + 16)

        self.setStyleSheet(f'''
            QSpinBox {{
                background-color: #121212;
                color: white;
                border: 1px solid #333333;
                border-radius: 2px;
                padding-left: 2px;
                padding-right: 2px;
            }}
            QSpinBox:hover {{
                border: 1px solid #444444;
            }}
            QSpinBox:focus {{
                border: 1px solid #4e8cff;
            }}

            QSpinBox::up-button, QSpinBox::down-button {{
                subcontrol-origin: border;
                width: 16px;
                background-color: #2a2a2a;
                border-left: 1px solid #333333;
            }}
            QSpinBox::up-button {{
                subcontrol-position: top right;
                border-top-right-radius: 2px;
            }}
            QSpinBox::down-button {{
                subcontrol-position: bottom right;
                border-bottom-right-radius: 2px;
            }}
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                background-color: #3a3a3a;
            }}
            QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {{
                background-color: #4a4a4a;
            }}
            QSpinBox::up-arrow, QSpinBox::down-arrow {{
                width: 8px;
                height: 8px;
            }}
        ''')

        self.setRange(range_min, range_max)
        self.setSingleStep(step)
        self.setValue(value)

    def setRange(self, a, b):  # type: ignore[override]
        super().setRange(a, b)


class DoubleSpinBoxWidget(QtWidgets.QDoubleSpinBox):

    def __init__(self, range_min, range_max, step, decimals, value):
        super().__init__()

        self.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Weight.Bold))
        self.setMinimumWidth(self.sizeHint().width())
        self.setMinimumHeight(self.sizeHint().height())
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        max_value_width = self.fontMetrics().horizontalAdvance(str(self.maximum()))
        self.setMaximumWidth(max_value_width + 16)

        self.setStyleSheet(f'''
            QDoubleSpinBox {{
                background-color: #121212;
                color: white;
                border: 1px solid #333333;
                border-radius: 2px;
                padding-left: 2px;
                padding-right: 2px;
            }}
            QDoubleSpinBox:hover {{ border: 1px solid #444444; }}
            QDoubleSpinBox:focus {{ border: 1px solid #4e8cff; }}

            QToolTip {{
                color: white;
                background-color: #2a2a2a;
                border: 1px solid #5a5a5a;
                padding: 4px;
                border-radius: 4px;
            }}

            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                width: 16px;
                background-color: #2a2a2a;
                border-left: 1px solid #333333;
            }}
            QDoubleSpinBox::up-button {{
                subcontrol-position: top right;
                border-top-right-radius: 2px;
            }}
            QDoubleSpinBox::down-button {{
                subcontrol-position: bottom right;
                border-bottom-right-radius: 2px;
            }}
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
                background-color: #3a3a3a;
            }}
            QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed {{
                background-color: #4a4a4a;
            }}
            QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {{
                width: 8px;
                height: 8px;
            }}
        ''')

        self.setDecimals(decimals)
        self.setRange(range_min, range_max)
        self.setSingleStep(step)
        self.setValue(value)

    def setRange(self, a, b):  # type: ignore[override]
        super().setRange(a, b)

    def setDecimals(self, d):  # type: ignore[override]
        super().setDecimals(d)


class PushButtonWidget(QtWidgets.QPushButton):
    def __init__(self, name='', checkable=False, enable=True, checked=False):
        super().__init__(name)
        self.setCheckable(checkable)
        self.setEnabled(enable)
        self.setChecked(checked)
        self.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Weight.Bold))
        self.setStyleSheet('''
            QPushButton {
                background-color: #121212;
                border-style: outset;
                border-radius: 2px;
                color: #FFFFFF;
                padding-left: 2px;
                padding-right: 2px;
            }
            QPushButton:hover {
                background-color: #4169e1;
            }
            QPushButton:pressed {
                background-color: #045c64;
                border-style: inset;
            }
            QPushButton:checked {
                background-color: #a52a2a;
                border-style: inset;
            }
        ''')
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        button_size = self.fontMetrics().boundingRect(self.text()).size()
        if button_size.width() < 100:
            button_size.setWidth(100)
        else:
            button_size.setWidth(button_size.width() + 24)
        if button_size.height() < 24:
            button_size.setHeight(24)
        else:
            button_size.setHeight(button_size.height() + 16)
        self.setFixedSize(button_size.width(), button_size.height())


class CheckBoxWidget(QtWidgets.QCheckBox):
    def __init__(self, name=''):
        super().__init__(name)
        self.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Weight.Bold))
        self.setStyleSheet('''
            QCheckBox {
                background-color: #121212;
                color: #FFFFFF;
                padding: 2px;
            }
            QCheckBox::indicator {
                width: 25px;
                height: 25px;
                border-radius: 4px;
                border: 2px solid #AAAAAA;
                background-color: #121212;
            }
            QCheckBox::indicator:checked {
                background-color: #4169e1;
                border: 2px solid #4169e1;
            }
        ''')
        self.setChecked(False)


class RadioButtonWidget(QtWidgets.QRadioButton):
    def __init__(self, name='', color="rgb(192, 255, 62)", autoex=False):
        super().__init__(name)
        self.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Weight.Bold))
        self.setAutoExclusive(autoex)
        self.setStyleSheet(f'''
            QRadioButton {{
                background-color: #1E1E1E;
                color: white;
            }}
            QRadioButton::indicator {{
                width: 8px;
                height: 8px;
            }}
            QRadioButton::indicator::unchecked {{
                border: 2px solid rgb(200, 200, 200);
                border-radius: 4px;
            }}
            QRadioButton::indicator::checked {{
                background-color: {color};
                border: 2px solid {color};
                border-radius: 4px;
            }}
        ''')


class ComboBoxWidget(QtWidgets.QComboBox):
    def __init__(self, list_items, length=128):
        super().__init__()
        for item in list_items:
            self.addItem(item)
        self.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Weight.Bold))
        self.setStyleSheet('''
            QComboBox {
                background-color: #121212;
                color: #FFFFFF;
                border: 1px solid #555555;
                border-radius: 2px;
                padding-left: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
            }
            QComboBox QAbstractItemView {
                background-color: #121212;
                color: #FFFFFF;
                selection-background-color: #4169e1;
            }
        ''')
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(24)
        self.setMaximumWidth(length)

class LineEditWidget(QtWidgets.QLineEdit):
    def __init__(self):
        super().__init__()
        self.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Weight.Bold))
        self.setStyleSheet('''
            QLineEdit {
                background-color: #444444;
                color: white;
            }
        ''')


class TextEditWidget(QtWidgets.QTextEdit):
    def __init__(self):
        super().__init__()
        self.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Weight.Bold))
        self.setStyleSheet('''
            QTextEdit {
                background-color: #444444;
                color: white;
                selection-background-color: #0096FF;
            }
        ''')


class SliderWidget(QtWidgets.QSlider):
    def __init__(self, mi, ma, value, tick=False):
        super().__init__(QtCore.Qt.Orientation.Horizontal)
        self.setMinimum(mi)
        self.setMaximum(ma)
        self.setSingleStep(1)
        self.setValue(value)
        if tick:
            self.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background-color: #222222;
                border: 1px solid #222222;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 8px;
                height: 16px;
                background-color: #FFFFFF;
                border: 1px solid #222222;
                border-radius: 6px;
                margin: -5px 0;
            }
            QSlider::sub-page:horizontal {
                height: 8px;
                background-color: #222222;
                border: 1px solid #222222;
                border-radius: 2px;
            }
        """)


class DialWidget(QtWidgets.QDial):
    def __init__(self, mi, ma, value):
        super().__init__()
        self.setMinimum(mi)
        self.setMaximum(ma)
        self.setValue(value)
        self.setStyleSheet('''
            QDial {
                background-color: #303030;
                border: none;
            }
            QDial::handle {
                background-color: #777777;
                border-radius: 6px;
            }
            QDial::handle:hover {
                background-color: #999999;
            }
            QDial::handle:pressed {
                background-color: #555555;
            }
        ''')


class DialogWidget(QtWidgets.QDialog):
    dialog_closed = QtCore.pyqtSignal()

    def __init__(self, interrupt=False):
        super().__init__()
        self.interrupt = interrupt
        self.setFixedSize(320, 64)
        self.setStyleSheet(''' 
            QDialog {
                background-color: #121212;
                color: #FFFFFF;
            }
            QLabel {
                color: #FFFFFF;
            }
        ''')
        self.setWindowTitle("Please Wait")

    def keyPressEvent(self, event):
        if self.interrupt:
            if event.key() == QtCore.Qt.Key.Key_Escape:
                self.dialog_closed.emit()
            else:
                event.ignore()
        else:
            event.ignore()


class MessageBoxWidget(QtWidgets.QMessageBox):
    def __init__(self, title, message):
        super().__init__()
        self.setWindowTitle(title)
        self.setText(message)
        self.setStandardButtons(QtWidgets.QMessageBox.StandardButton.NoButton)
        self.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        self.setStyleSheet("""
            QMessageBox {
                background-color: #121212;
                color: #EEEEEE;
                text-align: center;
            }
            QLabel {
                color: #EEEEEE;
            }
        """)


def create_dock(name=''):
    dock = DockWidget(name)
    group = GroupWidget()
    dock.setWidget(group)
    return dock, group


def create_scroll_area(layout="F"):
    scroll_area = ScrollAreaWidget()
    content_widget = QtWidgets.QWidget(scroll_area)
    content_widget.setStyleSheet("background-color: #232629;")
    scroll_area.setWidget(content_widget)
    if layout == "F":
        layout = QtWidgets.QFormLayout(content_widget)
    elif layout == "G":
        layout = QtWidgets.QGridLayout(content_widget)
    elif layout == "H":
        layout = QtWidgets.QHBoxLayout(content_widget)
    elif layout == "V":
        layout = QtWidgets.QVBoxLayout(content_widget)
    else:
        print("Invalid layout")
    content_widget.setLayout(layout)
    return scroll_area, layout


def create_dialog(labtex=False, interrupt=False):
    dialogue = DialogWidget(interrupt)
    layout = QtWidgets.QVBoxLayout()
    label = LabelWidget("Task is running, please wait...")
    layout.addWidget(label)
    dialogue.setLayout(layout)
    dialogue.setModal(True)
    if labtex:
        return dialogue, label
    else:
        return dialogue
