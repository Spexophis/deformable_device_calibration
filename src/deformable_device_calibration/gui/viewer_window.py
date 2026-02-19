# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QMutex, QMutexLocker, pyqtSlot, pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QHBoxLayout, QStackedWidget, QGridLayout

from . import custom_widgets as cw
from . import gl_viewer


class FramePool(QObject):
    def __init__(self, shape=(2048, 2048), dtype=np.uint16, n_buffers=4):
        super().__init__()
        self._buffers = [np.empty(shape, dtype=dtype) for _ in range(n_buffers)]
        self._free = list(range(n_buffers))
        self._in_use = set()
        self._m = QMutex()

    def acquire(self):
        """Reserve a buffer index for writing. Returns idx or None if none free."""
        with QMutexLocker(self._m):
            if not self._free:
                return None
            idx = self._free.pop()
            self._in_use.add(idx)
            return idx

    def buffer(self, idx: int) -> np.ndarray:
        return self._buffers[idx]

    @pyqtSlot(object)
    def release(self, token):
        """Return a buffer to the free list after viewer consumed/discarded it."""
        idx = int(token)
        with QMutexLocker(self._m):
            if idx in self._in_use:
                self._in_use.remove(idx)
                self._free.append(idx)


class PhotonPool(QObject):
    def __init__(self, max_len=2 ** 16, dt_s=4e-6, px=(64, 64), parent=None):
        super().__init__(parent)
        self.max_len = int(max_len)
        self.buf_0 = deque(np.zeros(self.max_len, dtype=np.int64), maxlen=self.max_len)
        self.buf_1 = deque(np.zeros(self.max_len, dtype=np.int64), maxlen=self.max_len)
        self.dt_s = dt_s
        self.xt = np.arange(self.max_len) * float(self.dt_s)
        self.img_0 = np.zeros(px, dtype=np.float64)
        self.img_1 = np.zeros(px, dtype=np.float64)

    def new_acquire(self, recon_img, counts, amps=None):
        self.buf_0.extend(counts[0])
        if amps is not None:
            self.buf_1.extend(amps)
        else:
            self.buf_1.extend(counts[1])
        self.img_0 = recon_img[0]
        self.img_1 = recon_img[1]

    def reset_buffer(self, max_len: int | None = None, dt_s:float | None = None, px:tuple | None = None):
        if max_len is not None:
            self.max_len = min(int(max_len), int(2 ** 16))
        self.buf_0 = deque(np.zeros(self.max_len, dtype=np.int64), maxlen=self.max_len)
        self.buf_1 = deque(np.zeros(self.max_len, dtype=np.int64), maxlen=self.max_len)
        if dt_s is not None:
            self.dt_s = float(dt_s)
        self.xt = np.arange(self.max_len) * float(self.dt_s)
        if px is not None:
            self.img_0 = np.zeros(px, dtype=np.float64)
            self.img_1 = np.zeros(px, dtype=np.float64)

class LiveViewer(QWidget):
    frame_idx_signal = pyqtSignal(int)

    def __init__(self, config, logg, parent=None):
        super().__init__(parent)
        self.config = config
        self.logg = logg
        pg.setConfigOptions(useOpenGL=True, antialias=False)
        self._setup_ui()
        self._overlay_n = 0
        self.h = 2048
        self.w = 2048
        self.pool = FramePool(shape=(self.h, self.w), dtype=np.uint16, n_buffers=4)
        self.photon_pool = PhotonPool()
        self.fft_mode = False
        self.fft_worker = None
        self.cxt = None
        self.data_curve_0 = None
        self.psr_mode = False
        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None
        self.wfr_mode = False
        self.wfr_worker = None
        self._setup_signal_connections()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Vertical)

        image_widget = QWidget()
        image_layout = self._create_image_widgets()
        image_widget.setLayout(image_layout)
        splitter.addWidget(image_widget)

        plot_widget = QWidget()
        plot_layout = self._create_plot_widgets()
        plot_widget.setLayout(plot_layout)
        splitter.addWidget(plot_widget)

        layout.addWidget(splitter)
        self.setLayout(layout)

    def _setup_signal_connections(self):
        self.QSlider_black.valueChanged.connect(self.on_black_change)
        self.QSlider_white.valueChanged.connect(self.on_white_change)
        self.QPushButton_contrast_auto.clicked.connect(self.auto_contrast)
        self.QPushButton_contrast_manual.clicked.connect(self.manual_contrast)
        self.image_viewer.mousePixelChanged.connect(self.on_mouse)
        self.image_viewer.frameConsumed.connect(self.pool.release, Qt.ConnectionType.QueuedConnection)
        self.image_viewer.frameDiscarded.connect(self.pool.release, Qt.ConnectionType.QueuedConnection)
        self.frame_idx_signal.connect(self.on_frame_idx, Qt.ConnectionType.QueuedConnection)

    def _create_image_widgets(self):
        layout_view = QVBoxLayout()
        layout_view.setContentsMargins(4, 4, 4, 4)

        self.image_viewer = gl_viewer.GLGray16Viewer(use_pbo=True)
        self.image_viewer.set_levels(0, 65535, 1.0)

        self.fft_viewer = gl_viewer.GLGray16Viewer(use_pbo=False)  # FFT frames (smaller; PBO not needed)

        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.image_viewer)  # index 0
        self.view_stack.addWidget(self.fft_viewer)  # index 1

        controls = QWidget()
        row = QHBoxLayout(controls)
        self.QComboBox_viewer_selection = cw.ComboBoxWidget(list_items=["Image", "FFT"])
        self.QSlider_black = cw.SliderWidget(0, 65535, 0)
        self.QSpinBox_black = cw.SpinBoxWidget(0, 65535, 1, 0)
        self.QSlider_white = cw.SliderWidget(0, 65535, 65535)
        self.QSpinBox_white = cw.SpinBoxWidget(0, 65535, 1, 65535)
        self.QPushButton_contrast_manual = cw.PushButtonWidget("Set")
        self.QPushButton_contrast_auto = cw.PushButtonWidget("Auto Set")

        row.addWidget(self.QComboBox_viewer_selection)
        row.addWidget(cw.LabelWidget("Min"))
        row.addWidget(cw.LabelWidget("0"))
        row.addWidget(self.QSlider_black)
        row.addWidget(self.QSpinBox_black)
        row.addWidget(cw.LabelWidget("Max"))
        row.addWidget(self.QSlider_white)
        row.addWidget(self.QSpinBox_white)
        row.addWidget(cw.LabelWidget("65535"))
        row.addWidget(self.QPushButton_contrast_manual)
        row.addWidget(self.QPushButton_contrast_auto)

        self.QLabel_cursor = cw.LabelWidget("x:-  y:-  v:-")

        layout_view.addWidget(controls)
        layout_view.addWidget(self.view_stack, stretch=1)
        layout_view.addWidget(self.QLabel_cursor)
        return layout_view

    def _create_plot_widgets(self):
        layout_plot = QGridLayout()

        self.graph_plot_0 = pg.PlotWidget()
        self.graph_plot_0.setAspectLocked(True)
        self.graph_plot_0.setLabel('left', 'Y Position', units='v')
        self.graph_plot_0.setLabel('bottom', 'X Position', units='v')
        self.graph_plot_0.getAxis('left').enableAutoSIPrefix(False)
        self.graph_plot_0.getAxis('bottom').enableAutoSIPrefix(False)

        self.graph_img_item_0 = pg.ImageItem(axisOrder="row-major")  # numpy (H,W)
        self.graph_plot_0.addItem(self.graph_img_item_0)
        self.graph_plot_0.invertY(True)

        self.data_plot_0 = pg.PlotWidget()
        self.data_plot_0.showGrid(x=True, y=True)

        self.data_curve_0 = self.data_plot_0.plot()

        self.data_curve_0.setDownsampling(auto=True, method="peak")
        self.data_curve_0.setSkipFiniteCheck(True)

        pi_0 = self.data_plot_0.getPlotItem()
        pi_0.setClipToView(True)
        pi_0.enableAutoRange(x=False)

        layout_plot.addWidget(self.graph_plot_0, 2, 0)
        layout_plot.addWidget(self.data_plot_0, 2, 1)
        return layout_plot

    def on_mouse(self, ix, iy, val):
        if ix < 0:
            self.QLabel_cursor.setText("x:-  y:-  v:-")
        else:
            self.QLabel_cursor.setText(f"x:{ix}  y:{iy}  v:{val}")

    def switch_camera(self, h, w):
        self.h, self.w = h, w
        self.pool = FramePool(shape=(self.h, self.w), dtype=np.uint16, n_buffers=4)
        self.image_viewer.frameConsumed.connect(self.pool.release, Qt.ConnectionType.QueuedConnection)
        self.image_viewer.frameDiscarded.connect(self.pool.release, Qt.ConnectionType.QueuedConnection)

    def on_camera_update_from_thread(self, frame: np.ndarray):
        """Runs in camera thread. Do NOT touch Qt widgets here."""
        if frame is None:
            return

        # normalize shape/dtype
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        if frame.dtype != np.uint16:
            frame = frame.astype(np.uint16, copy=False)

        idx = self.pool.acquire()
        if idx is None:
            return  # drop if GUI behind

        dst = self.pool.buffer(idx)
        np.copyto(dst, frame, casting="no")

        # send only index to GUI thread
        self.frame_idx_signal.emit(idx)

    @pyqtSlot(int)
    def on_black_change(self, value: int):
        self.QSpinBox_black.setValue(value)

    @pyqtSlot(int)
    def on_white_change(self, value: int):
        self.QSpinBox_white.setValue(value)

    @pyqtSlot()
    def manual_contrast(self):
        self.image_viewer.set_levels(self.QSpinBox_black.value(), self.QSpinBox_white.value())

    @pyqtSlot()
    def auto_contrast(self):
        b, w = self.image_viewer.auto_levels()
        self.QSlider_black.setValue(b)
        self.QSlider_white.setValue(w)

    @pyqtSlot(int)
    def on_frame_idx(self, idx: int):
        self.image_viewer.set_frame(self.pool.buffer(idx), token=idx)
        if self.wfr_mode:
            self.wfr_worker.push_frame(self.pool.buffer(idx))
        if self.fft_mode:
            self.fft_worker.push_frame(self.pool.buffer(idx))

    def on_fft_frame(self, frame_u16):
        self.fft_viewer.set_frame(frame_u16)

    def on_wfr_frame(self, frame_u16, levels=None):
        self.graph_img_item_0.setImage(frame_u16, autoLevels=(levels is None))
        if levels is not None:
            self.graph_img_item_0.setLevels(levels)
        # self.data_curve_0.setData(amp_lst)

    def set_plot_1(self, n):
        self.QComboBox_plot_selection.setCurrentIndex(n)

    def plot_trace(self, y, x=None, overlay=False):
        y = np.asarray(y)
        if y.size == 0:
            return
        if not overlay:
            self.data_plot_0.clear()
            self._overlay_n = 0
        if x is None:
            x = np.arange(y.size)
        self.data_plot_0.enableAutoRange(x=True)
        color = pg.intColor(self._overlay_n, hues=12)  # 12 distinct-ish hues, repeats after 12
        pen = pg.mkPen(color=color, width=1.)
        self._overlay_n += 1
        self.data_plot_0.plot(x, y, pen=pen)

    def stream_trace(self, x: np.ndarray, y_0: np.ndarray):
        """
        Update the 1D trace plot lively.
        """
        if y_0 is not None:
            self.data_plot_0.clear()
            self.data_curve_0 = self.data_plot_0.plot()
            self.data_curve_0.setDownsampling(auto=True, method="peak")
            self.data_curve_0.setSkipFiniteCheck(True)
            self.data_plot_0.enableAutoRange(x=True)
            self.data_curve_0.setData(x, y_0)

    def stream_trace_update(self, xt: np.ndarray, counts_0: np.ndarray):
        self.data_curve_0.setData(xt, counts_0)

    def set_graph_image(self, img_0: np.ndarray, levels=None):
        self.graph_img_item_0.setImage(img_0, autoLevels=(levels is None))
        if levels is not None:
            self.graph_img_item_0.setLevels(levels)

    def set_graph_with_axes(self, img_0: np.ndarray, x_axis=None, y_axis=None, levels=None):
        self.set_graph_image(img_0, levels)

        if x_axis is not None and y_axis is not None:
            self.x_min, self.x_max = x_axis[0], x_axis[-1]
            self.y_min, self.y_max = y_axis[0], y_axis[-1]

            h, w = img_0.shape
            pixel_width = (self.x_max - self.x_min) / w
            pixel_height = (self.y_max - self.y_min) / h
            self.graph_img_item_0.setRect(pg.QtCore.QRectF(self.x_min, self.y_min,
                                                           self.x_max - self.x_min,
                                                           self.y_max - self.y_min))
            self.graph_plot_0.setAspectLocked(True, ratio=pixel_height / pixel_width)
            self.graph_plot_0.setRange(xRange=[self.x_min, self.x_max],
                                       yRange=[self.y_min, self.y_max], padding=0)
        # else:
        #     self.graph_img_item_0.setRect(pg.QtCore.QRectF(self.x_min, self.y_min,
        #                                                    self.x_max - self.x_min,
        #                                                    self.y_max - self.y_min))
        #
        #     self.graph_plot_0.setRange(xRange=[self.x_min, self.x_max],
        #                                yRange=[self.y_min, self.y_max], padding=0)
