# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from deformable_device_calibration import run_threads
import numpy as np


class MockCamera:
    
    class CameraSettings:
        def __init__(self):
            self.t_clean = 0
            self.t_readout = 0
            self.t_exposure = 0
            self.t_accumulate = 0
            self.t_kinetic = 0
            self.gain = 0
            self.bin_h = 1
            self.bin_v = 1
            self.start_h = 0
            self.end_h = 2048
            self.start_v = 0
            self.end_v = 2048
            self.pixels_x = 2048
            self.pixels_y = 2048
            self.img_size = self.pixels_x * self.pixels_y
            self.ps = 3.45  # micron
            self.buffer_size = 4
            self.acq_num = 0
            self.acq_first = 0
            self.acq_last = 0
            self.valid_index = 0
            self.data = None
            
    def __init__(self):
        self._settings = self.CameraSettings()
        self.buffer_size = 4
        self.data = None
        self.acq_thread = None

    def __getattr__(self, item):
        if hasattr(self._settings, item):
            return getattr(self._settings, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def close(self):
        pass

    def prepare_live(self):
        self.data = run_threads.CameraDataList(self.buffer_size)
        self.acq_thread = run_threads.CameraAcquisitionThread(self)

    def start_live(self):
        self.acq_thread.start()

    def stop_live(self):
        if self.acq_thread is not None:
            self.acq_thread.stop()
            self.acq_thread = None
        self.data = None

    def prepare_acquisition(self, n):
        self.data = run_threads.CameraDataList(n)
        self.acq_thread = run_threads.CameraAcquisitionThread(self)

    def start_acquisition(self):
        self.acq_thread.start()

    def stop_acquisition(self):
        self.acq_thread.stop()
        self.acq_thread = None

    def get_images(self):
        self.data.add_element([np.random.randint(0, 2**14, size=(self.pixels_y, self.pixels_x), dtype=np.uint16)])

    def get_last_image(self):
        return np.random.randint(0, 2**14, size=(self.pixels_y, self.pixels_x), dtype=np.uint16)
