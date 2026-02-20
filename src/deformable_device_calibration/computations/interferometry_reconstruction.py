# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import json
import os
import time

import numpy as np
import tifffile as tf
from numpy.fft import fft2, ifft2, fftshift
from skimage.restoration import unwrap_phase

from deformable_device_calibration import logger
from deformable_device_calibration.utilities import image_processor as ipr
from deformable_device_calibration.utilities import zernike_generator as tz


class WavefrontSensing:

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self.fx_center: int = 816
        self.fy_center: int = 827
        self.half_nx: int = 128
        self.half_ny: int = 128
        self.radius: int = 128
        self.msk_hdl: bool = False
        self.wrp_hdl: bool = False
        self._meas = None
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
        self.fx_center = parameters[0]
        self.fy_center = parameters[1]
        self.half_nx = parameters[2]
        self.half_ny = parameters[3]
        self.radius = parameters[4]
        self.msk_hdl = parameters[5]
        self.wrp_hdl = parameters[6]

    def wavefront_reconstruction(self):
        imf = fftshift(fft2(self.meas))
        cf = imf[self.fy_center - self.half_ny: self.fy_center + self.half_ny, self.fx_center - self.half_nx:self.fx_center + self.half_nx]
        ph = ifft2(fftshift(cf))
        if self.msk_hdl:
            msk = self._elliptical_mask((self.radius, self.radius), cf.shape)
        else:
            msk = 1
        phase = np.arctan2(ph.imag, ph.real) * msk
        if self.wrp_hdl:
            self.wf = unwrap_phase(phase)
        else:
            self.wf = phase

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
    def _elliptical_mask(radius, size):
        coord_x = np.arange(0.5, size[0], 1.0)
        coord_y = np.arange(0.5, size[1], 1.0)
        y, x = np.meshgrid(coord_y, coord_x)
        x -= size[0] / 2.
        y -= size[1] / 2.
        return (x * x / (radius[0] * radius[0])) + (y * y / (radius[1] * radius[1])) <= 1

    def generate_influence_matrices(self, amp_list, data_folder, dm, sv=None, cfd=None, verbose=False):
        n_actuators, amp = dm.n_actuator, dm.amp
        dm.nly, dm.nlx = self.n_lenslets_y, self.n_lenslets_x
        dm.nls = self.n_lenslets_y * self.n_lenslets_x
        influence_matrix_phase = np.zeros((self.n_lenslets, n_actuators))
        wfs_phase = np.zeros((n_actuators, self.n_lenslets_y, self.n_lenslets_x))
        influence_matrix_zonal = np.zeros((2 * self.n_lenslets, n_actuators))
        influence_matrix_modal = np.zeros((dm.n_zernike, n_actuators))
        for filename in os.listdir(data_folder):
            if filename.endswith(".tif") & filename.startswith("actuator"):
                ind = int(filename.split("_")[1])
                if verbose:
                    self.logg.info(filename.split("_")[1])
                data_stack = tf.imread(os.path.join(data_folder, filename))
                n, x, y = data_stack.shape
                gdx = np.zeros((n - 1, self.n_lenslets_y, self.n_lenslets_x))
                gdy = np.zeros((n - 1, self.n_lenslets_y, self.n_lenslets_x))
                wf = np.zeros((n - 1, self.n_lenslets_y, self.n_lenslets_x))
                self.ref = data_stack[0]
                for i in range(1, n):
                    self.meas = data_stack[i]
                    gdx[i], gdy[i] = self.get_gradient_xy()
                    wf[i] = self.gradient_to_wavefront(gdx[i], gdy[i])
                # interaction_matrix =
                if "msk" not in locals():
                    image = np.sum(gdx, axis=0)
                    msk = image != 0
                    Z, dZdx, dZdy = tz.zernike_basis(dm.nlx, dm.nly, dm.nls, mask=msk, normalize_to="circle")
                    dm.zernike, dZdx_orth, dZdy_orth, T = tz.gs_orthogonalize(Z, msk, dZdx, dZdy)
                    dm.zslopes = np.zeros((2 * dm.nls, dm.n_zernike))
                    for j in range(dm.n_zernike):
                        dm.zslopes[:self.n_lenslets_x * self.n_lenslets_y, j] = dZdx_orth[j].flatten()
                        dm.zslopes[self.n_lenslets_x * self.n_lenslets_y:, j] = dZdy_orth[j].flatten()
                # phase
                mn = wfp.sum() / msk.sum()
                wfp = msk * (wfp - mn)
                mn = wfn.sum() / msk.sum()
                wfn = msk * (wfn - mn)
                wfg = (wfp - wfn) / (2 * amp)
                wfs_phase[ind] = wfg
                influence_matrix_phase[:, ind] = wfg.reshape(self.n_lenslets)
                # zonal
                influence_matrix_zonal[:self.n_lenslets, ind] = ((gdxp - gdxn) / (2 * amp)).reshape(self.n_lenslets)
                influence_matrix_zonal[self.n_lenslets:, ind] = ((gdyp - gdyn) / (2 * amp)).reshape(self.n_lenslets)
                # modal
                a1 = ipr.get_eigen_coefficients(np.concatenate((gdxp.flatten(), gdyp.flatten())), dm.zslopes, 32)
                a2 = ipr.get_eigen_coefficients(np.concatenate((gdxn.flatten(), gdyn.flatten())), dm.zslopes, 32)
                influence_matrix_modal[:, ind] = ((a1 - a2) / (2 * amp)).flatten()
        control_matrix_phase = ipr.pseudo_inverse(influence_matrix_phase, n=32)
        control_matrix_zonal = ipr.pseudo_inverse(influence_matrix_zonal, n=32)
        control_matrix_modal = ipr.pseudo_inverse(influence_matrix_modal, n=32)
        if sv is not None:
            fd = sv["Adaptive Optics"]["Deformable Mirror"][dm.dm_name]["Calibration File Folder"]
            t = time.strftime("%Y_%m_%d_%H_%M")
            fn = os.path.join(fd, f"influence_function_phase_{t}.tif")
            tf.imwrite(fn, influence_matrix_phase)
            fn = os.path.join(fd, f"control_matrix_phase_{t}.tif")
            tf.imwrite(fn, control_matrix_phase)
            dm.control_matrix_phase = control_matrix_phase
            sv["Adaptive Optics"]["Deformable Mirror"][dm.dm_name]["Phase Control Matrix"] = fn
            fn = os.path.join(fd, f"influence_function_images_{t}.tif")
            tf.imwrite(fn, wfs_phase)
            sv["Adaptive Optics"]["Deformable Mirror"][dm.dm_name]["Influence Function Images"] = fn
            fn = os.path.join(fd, f"influence_function_zonal_{t}.tif")
            tf.imwrite(fn, influence_matrix_zonal)
            fn = os.path.join(fd, f"control_matrix_zonal_{t}.tif")
            tf.imwrite(fn, control_matrix_zonal)
            dm.control_matrix_zonal = control_matrix_zonal
            sv["Adaptive Optics"]["Deformable Mirror"][dm.dm_name]["Zonal Control Matrix"] = fn
            fn = os.path.join(fd, f"influence_function_modal_{t}.tif")
            tf.imwrite(fn, influence_matrix_modal)
            fn = os.path.join(fd, f"control_matrix_modal_{t}.tif")
            tf.imwrite(fn, control_matrix_modal)
            dm.control_matrix_modal = control_matrix_modal
            sv["Adaptive Optics"]["Deformable Mirror"][dm.dm_name]["Modal Control Matrix"] = fn
            self.write_config(sv, cfd)

    @staticmethod
    def write_config(dataframe, dfd):
        with open(dfd, 'w') as f:
            json.dump(dataframe, f, indent=4)
