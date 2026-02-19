# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

from deformable_device_calibration import logger
from . import alpao_dm
from . import cobolt_laser
from . import flir_cmos


class DeviceManager:
    def __init__(self, config=None, logg=None, path=None, cf=None):
        self.config = config
        self.logg = logg or logger.setup_logging()
        self.data_folder = path
        self.cf = cf
        try:
            self.camera = flir_cmos.FLIRCamera(logg=self.logg)
        except Exception as e:
            from . import mock_cam
            self.camera = mock_cam.MockCamera()
            self.logg.error(f"{e}")
        try:
            self.laser = cobolt_laser.CoboltLaser(logg=self.logg, config=self.config)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.dfm = alpao_dm.DeformableMirror(logg=self.logg, config=self.config, path=self.data_folder, cfn=self.cf)
        except Exception as e:
            self.logg.error(f"{e}")
        self.logg.info("Finish initiating devices")

    def close(self):
        pass
        try:
            self.camera.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.laser.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.dfm.close()
        except Exception as e:
            self.logg.error(f"{e}")
