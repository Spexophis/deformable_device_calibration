# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import csv
import json
import os
import struct
import sys
import time

import numpy as np
import pandas as pd
import tifffile as tf

from deformable_device_calibration import logger
from deformable_device_calibration.computations import dynamic_controller as dc
from deformable_device_calibration.utilities import image_processor as ipr
from deformable_device_calibration.utilities import zernike_generator as tz

sys.path.append(r'C:\Program Files\Alpao\SDK\Samples\Python3')
if (8 * struct.calcsize("P")) == 32:
    from Lib.asdk import DM
else:
    from Lib64.asdk import DM


class DeformableMirror:

    def __init__(self, name="ALPAO", logg=None, config=None, path=None, cfn=None):
        self.dtp = path
        self.cfn = cfn
        self.logg = logg or logger.setup_logging()
        self.config = config or self.load_configs()
        self.dm_name = name
        self.dm_serial = self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Serial"]
        self.dm_model = self.dm_name + '_' + self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Model"]
        self.dm, self.n_actuator = self._initialize_dm(self.dm_serial)
        if self.dm is not None:
            self._configure_dm()
            self._get_zernike()
        else:
            raise RuntimeError(f"Error Initializing DM {self.dm_name}")
        # self.ctrl = dc.DynamicControl(n_states=self.n_zernike, n_inputs=self.n_zernike, n_outputs=self.n_zernike,
        #                               calib=self.ctrl_calib)
        self.g = 0.5
        try:
            self.set_dm(self.dm_cmd[self.current_cmd])
        except Exception as e:
            self.logg.error(f"Error set dm {e}")

    def __del__(self):
        pass

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    @staticmethod
    def load_configs():
        import json
        config_file = input("Enter configuration file directory: ")
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        return cfg

    def _initialize_dm(self, sn):
        try:
            dm = DM(sn)
            n_act = int(dm.Get('NBOfActuator'))
            self.logg.info("Number of actuator for " + sn + ": " + str(n_act))
            return dm, n_act
        except Exception as e:
            self.logg.error(f"Error Initializing DM {self.dm_name}: {e}")
            return None, None

    def _configure_dm(self):
        self.dm_cmd = [[0.] * self.n_actuator]
        try:
            self.control_matrix_phase = tf.imread(self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Phase Control Matrix"])
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")
        try:
            self.control_matrix_zonal = tf.imread(self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Zonal Control Matrix"])
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")
        try:
            self.control_matrix_modal = tf.imread(self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Modal Control Matrix"])
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")
        try:
            self.ctrl_calib = self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Control Calibration"]
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")
        try:
            self.read_cmd(self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Initial Flat"])
            self.current_cmd = 1
        except Exception as e:
            self.current_cmd = 0
            self.logg.error(f"Error Loading DM {self.dm_name} Initial Flat: {e}\n Started with Null")
        self.correction = []
        self.temp_cmd = []
        self.amp = 0.1

    def _get_zernike(self):
        try:
            influence_function_images = tf.imread(self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Influence Function Images"])
            nct, self.nly, self.nlx = influence_function_images.shape
            image = np.sum(influence_function_images, axis=0)
            msk = image != 0
            self.nls = self.nly * self.nlx
            self.n_zernike = tz.num_znk
            self.az = None
            Z, dZdx, dZdy = tz.zernike_basis(self.nlx, self.nly, self.n_zernike, mask=msk, normalize_to="circle")
            self.zernike, dZdx_orth, dZdy_orth, T = tz.gs_orthogonalize(Z, msk, dZdx, dZdy)
            self.zslopes = np.zeros((2 * self.nlx * self.nly, self.n_zernike))
            for j in range(self.n_zernike):
                self.zslopes[:self.nlx * self.nly, j] = dZdx_orth[j].flatten()
                self.zslopes[self.nlx * self.nly:, j] = dZdy_orth[j].flatten()
            # self.z2c = self._zernike_modes()
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")

    def _zernike_modes(self):
        """
        z2c index:
        0, 1 - tip / tilt
        2 - defocus
        3, 4 - astigmatism
        5, 6 - coma
        7, 8 - trefoil
        9 - spherical
        """
        pth = r"C:\Program Files\Alpao\SDK\Config"
        fn = f"{self.dm_serial}-Z2C.csv"
        Z2C = []
        with open(os.path.join(pth, fn), newline='') as csvfile:
            csvrows = csv.reader(csvfile, delimiter=' ')
            for row in csvrows:
                x = row[0].split(",")
                Z2C.append(x)
        for i in range(len(Z2C)):
            for j in range(len(Z2C[i])):
                Z2C[i][j] = float(Z2C[i][j])
        return Z2C

    def close(self):
        self.write_cmd(path=self.dtp, t=time.strftime("%Y%m%d%H%M%S") + '_')
        self.reset_dm()
        self.logg.info(f"DM {self.dm_name} Close")

    def reset_dm(self):
        self.dm.Reset()
        self.logg.info(f"DM {self.dm_name} Reset")

    def set_dm(self, values):
        if all(np.abs(v) < 1. for v in values):
            self.dm.Send(values)
            self.logg.info(f"DM {self.dm_name} set")
        else:
            raise ValueError("Some actuators exceed the DM push range!")

    def null_dm(self):
        self.dm.Send([0.] * self.n_actuator)
        self.logg.info(f"DM {self.dm_name} set to null")

    def get_correction(self, measurements, method="phase"):
        if method == 'phase':
            self.correction.append(list(self.g * self.amp * np.dot(self.control_matrix_phase, -measurements.reshape(self.nls))))
        else:
            gradx, grady = measurements
            measurement = np.concatenate((gradx.reshape(self.nls), grady.reshape(self.nls)))
            if method == 'zonal':
                self.correction.append(list(self.g * np.dot(self.control_matrix_zonal, -measurement)))
            elif method == 'modal':
                temp = self.get_zernike_coffs(gradx, grady)
                # a = np.zeros((self.n_zernike, 1))
                # a[:, 0] = -self.g * temp
                # _, u = self.ctrl.compute_control(a, False)
                self.correction.append(list(self.g * np.dot(self.control_matrix_modal, -temp)))
            else:
                self.logg.error(f"Invalid AO correction method")
                return
        _c = self.cmd_add(self.dm_cmd[self.current_cmd], self.correction[-1])
        self.dm_cmd.append(_c)

    def get_iterative_correction(self, measurements, method="modal"):
        if method == 'modal':
            gradx, grady = measurements
            temp = self.get_zernike_coffs(gradx, grady)
            a = np.zeros((self.n_zernike, 1))
            a[:, 0] = temp
            # _, u = self.ctrl.compute_control(a, False)
            self.correction.append(list(np.dot(self.control_matrix_modal, u[:, 0])))
            _c = self.cmd_add(self.dm_cmd[self.current_cmd], self.correction[-1])
            self.dm_cmd.append(_c)
        else:
            self.logg.error(f"Invalid AO correction method")
            return

    def get_zernike_cmd(self, j, a, method="modal"):
        if method == 'modal':
            a_s = a * ipr.get_eigen_coefficients(self.zslopes[:, j], self.zslopes, 14)
            return list(np.dot(self.control_matrix_modal, a_s))
        else:
            zerphs = a * self.zernike[j].reshape(self.nls)
            return list(np.dot(self.control_matrix_phase, zerphs))

    def get_zernike_coffs(self, gdx, gdy):
        return ipr.get_eigen_coefficients(np.concatenate((gdx.flatten(), gdy.flatten())), self.zslopes, 14)

    @staticmethod
    def cmd_add(cmd_0, cmd_1):
        return list(np.asarray(cmd_0) + np.asarray(cmd_1))

    def read_cmd(self, fnd):
        df = pd.read_excel(fnd, sheet_name=None)
        for key, cmd in df.items():
            self.dm_cmd.append(df[key]['Push'].tolist())

    def write_cmd(self, path, t, flatfile=False):
        if flatfile:
            filename = t + f"{self.dm_serial}_flat_file.xlsx"
            df = pd.DataFrame(self.dm_cmd[-1], index=np.arange(self.n_actuator), columns=['Push'])
            fd = os.path.join(path, filename)
            df.to_excel(str(fd), index_label='Actuator')
        else:
            filename = t + f"{self.dm_serial}_cmd_file.xlsx"
            fd = os.path.join(path, filename)
            data = {f'cmd{i}': cmd for i, cmd in enumerate(self.dm_cmd)}
            with pd.ExcelWriter(str(fd), engine='xlsxwriter') as writer:
                for sheet_name, list_data in data.items():
                    df = pd.DataFrame(list_data, index=np.arange(self.n_actuator), columns=['Push'])
                    df.to_excel(writer, sheet_name=sheet_name, index_label='Actuator')

    def write_flat_cmd(self, t, cmd):
        path = self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Calibration File Folder"]
        filename = f"flat_file_{self.dm_serial}_{t}.xlsx"
        fd = os.path.join(path, filename)
        df = pd.DataFrame(cmd, index=np.arange(self.n_actuator), columns=['Push'])
        df.to_excel(str(fd), index_label='Actuator')
        self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Initial Flat"] = str(fd)
        with open(self.cfn, 'w') as f:
            json.dump(self.config, f, indent=4)

    def save_sensorless_results(self, fd, a, v, p):
        df1 = pd.DataFrame(v, index=a, columns=['Values'])
        df2 = pd.DataFrame(p, index=np.arange(self.n_zernike), columns=['Amplitudes'])
        with pd.ExcelWriter(fd, engine='xlsxwriter') as writer:
            df1.to_excel(writer, sheet_name='Metric Values')
            df2.to_excel(writer, sheet_name='Peaks')

    def wavefront_zernike_reconstruct(self, wf, size=None, exclusive=True):
        return self.wavefront_recomposition(self.wavefront_decomposition(wf), size=size, exclusive=exclusive)

    def wavefront_decomposition(self, wf):
        za = np.zeros(self.n_zernike)
        for i in range(self.n_zernike):
            wz = self.zernike[i]
            za[i] = (wf * wz.conj()).sum() / (wz * wz.conj()).sum()
        return za

    def wavefront_recomposition(self, za, size=None, exclusive=True):
        if size is None:
            ny = self.nly
            nx = self.nlx
        else:
            ny, nx = size
        wf = np.zeros((ny, nx))
        if exclusive:
            za[:4] = 0
        for i in range(self.n_zernike):
            wf += za[i] * self.zernike[i]
        return wf
