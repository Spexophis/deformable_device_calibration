# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot


class CameraAcquisitionThread(threading.Thread):
    def __init__(self, cam, interval=0.001):
        super().__init__()
        self.cam = cam
        self.running = False
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.interval = interval

    def run(self):
        self.running = True
        while self.running:
            with self.condition:
                self.condition.wait(timeout=self.interval)

                if not self.running:
                    break

                # Acquire images while holding the lock
                self.cam.get_images()

    def stop(self):
        with self.condition:
            self.running = False
            self.condition.notify()  # Wake up thread immediately
        self.join()

    def trigger(self):
        """Manually trigger an immediate acquisition"""
        with self.condition:
            self.condition.notify()


class CameraDataList:

    def __init__(self, max_length):
        self.data_list = deque(maxlen=max_length)
        self.ind_list = deque(maxlen=max_length)
        self.callback = None
        self._lock = threading.Lock()

    def add_element(self, elements, ids=None):
        with self._lock:
            self.data_list.extend(elements)
            if ids is not None:
                self.ind_list.extend(list(range(ids[0], ids[1] + 1)))
            last = self.data_list[-1] if self.data_list else None
        if self.callback is not None and last is not None:
            self.callback(last)  # passes ndarray

    def get_elements(self):
        return np.array(self.data_list) if self.data_list else None

    def get_last_element(self, copy=False):
        with self._lock:
            if not self.data_list:
                return None
            arr = self.data_list[-1]
        return arr.copy() if copy else arr  # no copy for display

    def on_update(self, callback):
        self.callback = callback


class WFRWorker(QThread):
    wfr_ready = pyqtSignal(object)

    def __init__(self, fps=10, op=None, parent=None):
        super().__init__(parent)
        self.fps = float(fps)
        self.op = op
        self._running = True
        self._lock = threading.Lock()

    def stop(self):
        """Stop worker thread gracefully"""
        self._running = False

        if not self.wait(2000):  # 2 second timeout
            self.terminate()  # Force terminate if hung
            self.wait(1000)

    def push_frame(self, frame_u16: np.ndarray):
        if not self._running or frame_u16 is None or frame_u16.ndim != 2:
            return

        with self._lock:
            self.op.meas = np.array(frame_u16, copy=True)

    def run(self):
        period = 1.0 / max(self.fps, 0.1)
        next_t = time.perf_counter()

        try:
            while self._running:
                now = time.perf_counter()
                if now < next_t:
                    self.msleep(int((next_t - now) * 1000))
                    continue
                next_t = now + period

                with self._lock:
                    if self.op.meas is None:
                        continue

                # Process without holding lock
                self.op.wavefront_reconstruction()

                if self._running:
                    self.wfr_ready.emit(self.op.wf)

        except Exception as e:
            import logging
            logging.error(f"WFRWorker error: {e}")


@dataclass
class WFSLoopConfig:
    gain:               float = 0.5    # integrator gain  (0 < g ≤ 1)
    n_iterations:       int   = 32
    convergence_rms:    float = 0.05   # stop when WF RMS < this [rad]
    v_min:              float = -1.0
    v_max:              float =  1.0
    leaky_gain:         float = 1.0    # set < 1 to add leaky integration


@dataclass
class WFSLoopResult:
    voltages:       np.ndarray
    rms_history:    list[float] = field(default_factory=list)
    pv_history:     list[float] = field(default_factory=list)
    strehl_history:     list[float] = field(default_factory=list)
    converged:      bool = False
    n_iterations:   int  = 0


class TaskWorker(QThread):
    error = pyqtSignal(tuple)

    def __init__(self, task=None, n=1, parent=None):
        super().__init__(parent)
        self.task = task if task is not None else self._do_nothing
        self.n = n

    def run(self):
        try:
            for i in range(self.n):
                self.task()
        except Exception as e:
            self.error.emit((e, traceback.format_exc()))

    @pyqtSlot()
    def _do(self):
        self.task()

    @staticmethod
    def _do_nothing():
        pass
