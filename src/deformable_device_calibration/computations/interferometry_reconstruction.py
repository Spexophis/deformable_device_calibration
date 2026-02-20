# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import json
import os
import time

import numpy as np
import tifffile as tf
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
from skimage.restoration import unwrap_phase

from deformable_device_calibration import logger
from deformable_device_calibration.utilities import image_processor as ipr
from deformable_device_calibration.utilities import zernike_generator as tz


class WavefrontSensing:

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self.fx_center: int = 816
        self.fy_center: int = 827
        self.half_nx: int = 160
        self.half_ny: int = 160
        self.radius: int = 156
        self.msk_hdl: bool = False
        self.wrp_hdl: bool = False
        self.remove_tilt: bool = True
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
        if self._meas is not None:

            imf = fftshift(fft2(self._meas))
            self.auto_detect_carrier(imf, fit_method='gaussian')
            cf = imf[self.fy_center - self.half_ny: self.fy_center + self.half_ny,
            self.fx_center - self.half_nx: self.fx_center + self.half_nx]
            ph = ifft2(ifftshift(cf))

            if self.msk_hdl:
                msk = self._elliptical_mask((self.radius, self.radius), ph.shape)
            else:
                msk = 1

            phase = np.arctan2(ph.imag, ph.real) * msk

            if self.wrp_hdl:
                wf = unwrap_phase(phase)
            else:
                wf = phase

            if self.remove_tilt and self.wrp_hdl:
                wf = self._remove_plane(wf)

            self.wf = wf * msk

    def auto_detect_carrier(self, imf, dc_exclusion_radius: int = 80, search_radius: int = 150, fit_method: str = 'gaussian'):
        """
        Automatically locate the 1st-order carrier frequency in the FFT spectrum.
        """

        mag = np.abs(imf)
        ny, nx = mag.shape
        cy, cx = ny // 2, nx // 2

        # Block DC
        mag_search = mag.copy()
        mag_search[cy - dc_exclusion_radius: cy + dc_exclusion_radius,
        cx - dc_exclusion_radius: cx + dc_exclusion_radius] = 0

        # Initial guess: brightest surviving pixel
        peak_fy, peak_fx = np.unravel_index(np.argmax(mag_search), mag_search.shape)

        if fit_method == 'max':
            fy_c, fx_c = int(peak_fy), int(peak_fx)

        else:
            r = search_radius
            fy0, fy1 = max(0, peak_fy - r), min(ny, peak_fy + r)
            fx0, fx1 = max(0, peak_fx - r), min(nx, peak_fx + r)
            region = mag[fy0:fy1, fx0:fx1]
            log_region = np.log1p(region)

            if fit_method == 'centroid':
                thresh = region.max() * 0.1
                masked = np.where(region > thresh, region, 0)
                com = center_of_mass(masked)
                fy_c = int(round(com[0] + fy0))
                fx_c = int(round(com[1] + fx0))

            else:  # 'gaussian' (default)
                H, W = log_region.shape
                Y_g, X_g = np.mgrid[0:H, 0:W]
                # Initial estimates from centroid
                com = center_of_mass(log_region)

                def gauss2d(xy, amp, x0, y0, sx, sy, offset):
                    x, y = xy
                    return (offset
                            + amp * np.exp(-((x - x0) ** 2 / (2 * sx ** 2)
                                             + (y - y0) ** 2 / (2 * sy ** 2))))

                try:
                    p0 = [log_region.max() - log_region.min(),
                          com[1], com[0], 30, 30, log_region.min()]
                    popt, _ = curve_fit(
                        gauss2d,
                        (X_g.ravel(), Y_g.ravel()),
                        log_region.ravel(),
                        p0=p0, maxfev=10000)
                    fx_c = int(round(popt[1] + fx0))
                    fy_c = int(round(popt[2] + fy0))
                except RuntimeError:
                    # Fall back to centroid if Gaussian fit fails
                    thresh = region.max() * 0.1
                    masked = np.where(region > thresh, region, 0)
                    com = center_of_mass(masked)
                    fy_c = int(round(com[0] + fy0))
                    fx_c = int(round(com[1] + fx0))

        self.fy_center = fy_c
        self.fx_center = fx_c

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
    def _elliptical_mask(radii, shape):
        ry, rx = radii
        ny, nx = shape
        yv, xv = np.ogrid[-ny // 2: ny // 2, -nx // 2: nx // 2]
        return ((xv / rx) ** 2 + (yv / ry) ** 2 <= 1).astype(float)

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
        n_actuators, amp = dm.n_actuator, 0.08
        ny = self.half_ny * 2
        nx = self.half_nx * 2
        nxy = ny * nx
        influence_matrix_phase = np.zeros((nxy, n_actuators))
        wfs_phase = np.zeros((n_actuators, ny, nx))
        for filename in os.listdir(data_folder):
            if filename.endswith(".tif") & filename.startswith("actuator"):
                ind = int(filename.split("_")[1])
                if verbose:
                    self.logg.info(filename.split("_")[1])
                data_stack = tf.imread(os.path.join(data_folder, filename))
                imf = fftshift(fft2(np.average(data_stack, axis=0)))
                self.auto_detect_carrier(imf, fit_method='gaussian')
                wfs_phase[ind] = self._complex_differential_phase(data_stack[2], data_stack[1])
                influence_matrix_phase[:, ind] = wfs_phase[ind].ravel() / (2.0 * amp)
        # control_matrix_phase = ipr.pseudo_inverse(influence_matrix_phase, n=32)
        if sv is not None:
            fd = sv["Adaptive Optics"]["Deformable Mirror"][dm.dm_name]["Calibration File Folder"]
            t = time.strftime("%Y_%m_%d_%H_%M")
            fn = os.path.join(fd, f"interferometry_influence_function_phase_{t}.tif")
            tf.imwrite(fn, influence_matrix_phase)
            # fn = os.path.join(fd, f"interferometry_control_matrix_phase_{t}.tif")
            # tf.imwrite(fn, control_matrix_phase)
            # dm.control_matrix_phase = control_matrix_phase
            # sv["Adaptive Optics"]["Deformable Mirror"][dm.dm_name]["Phase Control Matrix"] = fn
            fn = os.path.join(fd, f"interferometry_influence_function_images_{t}.tif")
            tf.imwrite(fn, wfs_phase)
            self.write_config(sv, cfd)

    def _complex_differential_phase(self, raw_plus, raw_minus):
        """
        Compute the differential phase (plus - minus) in the complex domain
        to voids 2pi jump artifacts from wrapping boundary shifts.

        Valid as long as the poke response < pi rad everywhere,
        i.e. keep poke_amplitude small enough.
        """
        cf_plus = self._extract_carrier(raw_plus)
        cf_minus = self._extract_carrier(raw_minus)

        # Differential complex field: phase = arg(A * conj(B)) = phi_A - phi_B
        diff_field = ifft2(ifftshift(cf_plus * np.conj(cf_minus)))

        # This wrapped phase IS the true difference if |diff| < pi everywhere
        diff_phase = np.arctan2(diff_field.imag, diff_field.real)
        return diff_phase

    def _extract_carrier(self, raw_frame):
        """Return the extracted complex carrier (before IFFT), not the phase."""
        imf = fftshift(fft2(raw_frame.astype(float)))
        return imf[self.fy_center - self.half_ny: self.fy_center + self.half_ny,
        self.fx_center - self.half_nx: self.fx_center + self.half_nx]

    @staticmethod
    def write_config(dataframe, dfd):
        with open(dfd, 'w') as f:
            json.dump(dataframe, f, indent=4)
