# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import json
import os
import struct
import sys
import time

import numpy as np
import pandas as pd
import tifffile as tf

from deformable_device_calibration import logger
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
        self.dm_cmd = []
        try:
            self.control_matrix_phase = tf.imread(
                self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Phase Control Matrix"])
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")
        try:
            self.control_matrix_zonal = tf.imread(
                self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Zonal Control Matrix"])
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")
        try:
            self.control_matrix_modal = tf.imread(
                self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Modal Control Matrix"])
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")
        try:
            self.ctrl_calib = self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Control Calibration"]
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")
        try:
            self.read_cmd(self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Mirror Flat"])
        except Exception as e:
            self.dm_cmd = [[0.] * self.n_actuator]
            self.logg.error(f"Error Loading DM {self.dm_name} Mirror Flat: {e}\n")
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
                if j == 0:
                    self.zslopes[:self.nls, j] = dZdx_orth[j].flatten()
                    self.zslopes[self.nls:, j] = dZdy_orth[j].flatten()
                else:
                    self.zslopes[:self.nls, j] = (dZdx_orth[j] / np.std(dZdx_orth[j])).flatten()
                    self.zslopes[self.nls:, j] = (dZdy_orth[j] / np.std(dZdy_orth[j])).flatten()
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")

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

    def get_int_correction(self, measurement):
        delta_v = self.control_matrix_phase @ measurement.ravel()

    def get_sh_correction(self, measurements, method="phase"):
        gradx, grady = measurements
        measurement = np.concatenate((gradx.reshape(self.nls), grady.reshape(self.nls)))
        if method == 'zonal':
            self.correction.append(list(self.g * np.dot(self.control_matrix_zonal, -measurement)))
        # elif method == 'modal':
        #     temp = self.get_zernike_coffs(gradx, grady)
        #     self.correction.append(list(self.g * np.dot(self.control_matrix_modal, -temp)))
        else:
            self.logg.error(f"Invalid AO correction method")
            return
        _c = self.cmd_add(self.dm_cmd[self.current_cmd], self.correction[-1])
        self.dm_cmd.append(_c)

    def get_zernike_cmd(self, j, a, method="modal"):
        if method == 'modal':
            voltages = self.control_matrix_modal[:, j] * a
            return voltages.tolist()
        if method == 'zonal':
            target = self.zslopes[:, j] * a
            voltages = self.control_matrix_zonal @ target
            return voltages.tolist()
        return None

    def get_zernike_coffs(self, gdx, gdy):
        return ipr.get_eigen_coefficients(np.concatenate((gdx.flatten(), gdy.flatten())), self.zslopes, int(self.n_zernike * 0.9))

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
