# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import threading
import time
import traceback
from collections import deque

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


class MPDCountThread(threading.Thread):

    def __init__(self, daq, ind, interval=0.004):
        threading.Thread.__init__(self)
        self.daq = daq
        self.ind = ind
        self.interval = interval
        self.started = False
        self.running = False
        self.paused = False
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def run(self):
        self.started = True
        self.running = True
        while self.running:
            with self.condition:

                while self.paused and self.running:
                    self.condition.wait()

                if not self.running:
                    break

                self.condition.wait(timeout=self.interval)

                if not self.running:
                    break

                if not self.paused and self.running:
                    self.daq.get_photon_counts(self.ind)

    def pause(self):
        with self.condition:
            self.paused = True

    def resume(self):
        with self.condition:
            self.paused = False
            self.condition.notify()

    def stop(self):
        with self.condition:
            self.running = False
            self.paused = False
            self.condition.notify()
        self.join()

    def trigger(self):
        """Manually trigger an immediate count"""
        with self.condition:
            self.condition.notify()


class MPDCountList:

    def __init__(self, max_length):
        self.data_lists = [deque(maxlen=max_length), deque(maxlen=max_length)]
        for data_list in self.data_lists:
            data_list.extend([0])
        self.count_lists = [deque(maxlen=max_length), deque(maxlen=max_length)]
        self.ind_lists = [deque(maxlen=max_length), deque(maxlen=max_length)]
        self.count_starts = [1, 1]
        self.count_ends = [1, 1]
        self.count_lens = [max_length, max_length]
        self.callback = None
        self.request = None
        self.lock = threading.Lock()

    def add_element(self, elements: list, num: int, ind: int):
        with self.lock:
            d = np.asarray(elements, dtype=np.int64)
            self.count_starts[ind] = self.count_ends[ind] % self.count_lens[ind]
            self.count_ends[ind] = (self.count_ends[ind] + num) % self.count_lens[ind]
            counts = np.diff(np.insert(d, 0, self.data_lists[ind][-1]))
            self.count_lists[ind].extend(counts.tolist())
            self.data_lists[ind].extend(elements)
            if self.count_starts[ind] <= self.count_ends[ind]:
                indices = np.arange(self.count_starts[ind], self.count_ends[ind])
            else:
                indices = np.concatenate(
                    (np.arange(self.count_starts[ind], self.count_lens[ind]), np.arange(self.count_ends[ind])))
            self.ind_lists[ind].extend(indices.tolist())
            if self.callback is not None:
                self.callback(counts, list(indices), ind)

    def get_last_element(self, ind):
        c = self.data_lists[ind][-1] if self.data_lists[ind][-1] else None
        return c

    def get_elements(self, ind):
        r = np.array(self.data_lists[ind]) if self.data_lists[ind] else None
        d = self.count_lists[ind] if self.count_lists[ind] else None
        return r, d

    def on_update(self, callback):
        self.callback = callback

class PMTAmpThread(threading.Thread):

    def __init__(self, daq, interval=0.004):
        threading.Thread.__init__(self)
        self.daq = daq
        self.interval = interval
        self.running = False
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def run(self):
        self.running = True
        while self.running:
            with self.condition:
                self.condition.wait(timeout=self.interval)

                if not self.running:
                    break

                self.daq.get_pmt_amps()

    def stop(self):
        with self.condition:
            self.running = False
            self.condition.notify()  # Wake up immediately
        self.join()

    def trigger(self):
        """Manually trigger an immediate count"""
        with self.condition:
            self.condition.notify()


class PMTAmpList:

    def __init__(self, max_length):
        self.data_list = deque(maxlen=max_length)
        self.ind_list = deque(maxlen=max_length)
        self.count_len = max_length
        self.count_starts = 1
        self.count_ends = 1
        self.callback = None
        self.request = None
        self.lock = threading.Lock()

    def add_element(self, elements: list, num: int):
        with self.lock:
            self.count_starts = self.count_ends % self.count_len
            self.count_ends = (self.count_ends + num) % self.count_len
            self.data_list.extend(elements)
            if self.count_starts <= self.count_ends:
                indices = list(range(self.count_starts, self.count_ends))
            else:
                indices = list(range(self.count_starts, self.count_len))
                indices.extend(list(range(self.count_ends)))
            self.ind_list.extend(indices)
            if self.callback is not None:
                self.callback(np.array(elements), indices, 1)

    def get_elements(self):
        return np.array(self.data_list) if self.data_list else None

    def on_update(self, callback):
        self.callback = callback


class PSLiveWorker(QThread):
    psr_ready = pyqtSignal(object, object, object)
    psr_new = pyqtSignal()

    def __init__(self, reco, mpd_dat=None, pmt_dat=None, fps=10, parent=None):
        super().__init__(parent)
        self.mpd_dat = mpd_dat
        self.pmt_dat = pmt_dat
        self.reco = reco
        self.period_ms = max(1, int(1000 / max(float(fps), 0.1)))
        self._running = True
        self._lock = threading.Lock()

    def stop(self):
        """Stop worker thread gracefully"""
        self._running = False

        if not self.wait(2000):
            self.terminate()
            self.wait(1000)

        self.clear_data()

    def clear_data(self):
        """Release references to large data objects"""
        with self._lock:
            self.mpd_dat = None
            self.pmt_dat = None
            self.reco = None

    def run(self):
        try:
            while self._running:
                self.msleep(self.period_ms)
                with self._lock:
                    if self.mpd_dat is None or self.reco is None:
                        break
                    img_copy = self.reco.live_rec
                    counts_copy = self.mpd_dat.count_lists
                    if self.pmt_dat is not None:
                        amp_copy = self.pmt_dat.data_list
                    else:
                        amp_copy = None
                self.psr_ready.emit(img_copy, counts_copy, amp_copy)
                self.psr_new.emit()
        except Exception as e:
            import logging
            logging.error(f"PSLiveWorker error: {e}")
        finally:
            self.clear_data()


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
