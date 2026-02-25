# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import json
import os
import time

import numpy as np
import tifffile as tf

from deformable_device_calibration.utilities import image_processor as ipr
from deformable_device_calibration import logger
from . import actuator_response as ar


class WavefrontSensing:

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self.fx_center: int | None = None
        self.fy_center: int | None = None
        self.half_nx: int = 128
        self.half_ny: int = 128
        self.radius: int = 124
        self.msk_hdl: bool = False
        self.wrp_hdl: bool = False
        self.remove_tilt: bool = True
        self._meas = None
        self.tracker = None
        self.wf = None

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    @property
    def meas(self):
        return self._meas

    @meas.setter
    def meas(self, new_meas):
        self._meas = new_meas

    def update_parameters(self, parameters):
        self.half_nx = parameters[2]
        self.half_ny = parameters[3]
        self.radius = parameters[4]
        self.msk_hdl = parameters[5]
        self.wrp_hdl = parameters[6]

    def prepare_int_reconstruction(self):
        self.tracker = None

    def wavefront_reconstruction(self):
        if self._meas is not None:

            if self.tracker is None:
                self.tracker = ar.LiveCarrierTracker(image_shape=self._meas.shape,
                                                     seed_frames=5,
                                                     reject_radius_init=30.0,
                                                     reject_radius_min=8.0,
                                                     conjugate_radius=20.0,
                                                     ema_alpha=0.05,
                                                     )

            self.fy_center, self.fx_center = self.tracker.update(self._meas)

            if self._meas.ndim == 2:
                stack = self._meas[np.newaxis]
            else:
                stack = self._meas

            result = ar.extract_response(stack, fy=self.fy_center, fx=self.fx_center, half_n=self.half_nx)

            if self.wrp_hdl:
                wf = result["phase_wrapped"]
            else:
                wf = result["phase"]

            if self.msk_hdl:
                msk = self._mask((self.radius, self.radius), wf.shape)
            else:
                msk = 1

            self.wf = wf * msk

    def compute_wavefront(self, measurements):
        if measurements.ndim == 2:
            ny, nx = measurements.shape
            stack = measurements[np.newaxis]
        else:
            _, ny, nx = measurements.shape
            stack = measurements

        result = ar.extract_response(stack, fy=self.fy_center, fx=self.fx_center, half_n=self.half_nx)

        if self.wrp_hdl:
            wf = result["phase_wrapped"]
        else:
            wf = result["phase"]

        if self.msk_hdl:
            msk = self._mask((self.radius, self.radius), wf.shape)
        else:
            msk = 1

        return wf * msk

    def save_wfs_results(self, file_name, dm):
        try:
            tf.imwrite(file_name + f'_{dm.dm_serial}_int_wfs_raw.tif', self.meas)
        except Exception as e:
            self.logg.error(f"Error saving wfs raw image: {e}")
        try:
            tf.imwrite(file_name + f'_{dm.dm_serial}_int_recon_wf.tif', self.wf)
        except Exception as e:
            self.logg.error(f"Error saving wfs wavefront: {e}")

    @staticmethod
    def _mask(radii, shape):
        ry, rx = radii
        ny, nx = shape
        y, x = np.ogrid[:ny, :nx]
        cx = nx / 2
        cy = ny / 2
        distance_squared = (x - cx) ** 2 + (y - cy) ** 2
        return distance_squared <= rx ** 2

    @staticmethod
    def _remove_plane(wf):
        """Subtract a least-squares best-fit plane (piston + tip + tilt)."""
        ny, nx = wf.shape
        yv, xv = np.mgrid[0:ny, 0:nx]
        a = np.stack([np.ones(ny * nx), xv.ravel(), yv.ravel()], axis=1)
        coef, _, _, _ = np.linalg.lstsq(a, wf.ravel(), rcond=None)
        plane = coef[0] + coef[1] * xv + coef[2] * yv
        return wf - plane

    def generate_influence_matrices(self, amp_list, data_folder, dm, sv=None, cfd=None, verbose=False):
        n_actuators, amp = dm.n_actuator, dm.amp
        ny = self.half_ny * 2
        nx = self.half_nx * 2
        nxy = ny * nx
        influence_matrix_phase = np.zeros((nxy, n_actuators))
        wfs_phase = np.zeros((n_actuators, ny, nx))
        msk = self._mask((self.radius, self.radius), (ny, nx))
        for filename in os.listdir(data_folder):
            if filename.endswith(".tif") & filename.startswith("actuator"):
                ind = int(filename.split("_")[1])
                if verbose:
                    self.logg.info(filename.split("_")[1])
                data_stack = tf.imread(os.path.join(data_folder, filename))
                nz, _, _ = data_stack.shape
                hn = nz // 2
                frames_minus, frames_plus = data_stack[:hn], data_stack[hn:]
                pp = ar.process_push_pull(frames_plus, frames_minus, amp, self.fy_center, self.fx_center, self.half_nx)
                wfs_phase[ind] = msk * pp["influence"]
                influence_matrix_phase[:, ind] = wfs_phase[ind].ravel()
        control_matrix_phase = ipr.pseudo_inverse(influence_matrix_phase, n_modes_kept=72)
        if sv is not None:
            fd = sv["Adaptive Optics"]["Deformable Mirror"][dm.dm_name]["Calibration File Folder"]
            t = time.strftime("%Y_%m_%d_%H_%M")
            fn = os.path.join(fd, f"interferometry_influence_function_phase_{t}.tif")
            tf.imwrite(fn, influence_matrix_phase)
            fn = os.path.join(fd, f"interferometry_control_matrix_phase_{t}.tif")
            tf.imwrite(fn, control_matrix_phase)
            dm.control_matrix_phase = control_matrix_phase
            sv["Adaptive Optics"]["Deformable Mirror"][dm.dm_name]["Phase Control Matrix"] = fn
            fn = os.path.join(fd, f"interferometry_influence_function_images_{t}.tif")
            tf.imwrite(fn, wfs_phase)
            self.write_config(sv, cfd)

    @staticmethod
    def write_config(dataframe, dfd):
        with open(dfd, 'w') as f:
            json.dump(dataframe, f, indent=4)
