# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from deformable_device_calibration import logger
from . import shwfs_reconstruction, interferometry_reconstruction


class ComputationManager:
    def __init__(self, config=None, logg=None, path=None):
        self.config = config
        self.logg = logg or logger.setup_logging()
        self.data_folder = path
        self.shwfr = shwfs_reconstruction.WavefrontSensing(logg=self.logg)
        self.intwfr = interferometry_reconstruction.WavefrontSensing(logg=self.logg)

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging
