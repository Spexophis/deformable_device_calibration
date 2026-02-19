# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import pycobolt

from deformable_device_calibration import logger


class CoboltLaser:

    def __init__(self, logg=None, config=None):
        self.logg = logg or logger.setup_logging()
        self.config = config or self.load_configs()
        laser_dict = {}
        for las, inf in self.config["Light Sources"]["Lasers"]["Cobolt"].items():
            laser_dict[las] = inf["Serial"]
        self.lasers, self._h = self._initiate_lasers(laser_dict)

    def _initiate_lasers(self, laser_dict):
        lasers = {}
        for laser, ids in laser_dict.items():
            try:
                lasers[laser] = pycobolt.Cobolt06MLD(serialnumber=ids)
                lasers[laser].send_cmd('@cobas 0')
                self.logg.info("{} Laser Connected".format(laser))
            except Exception as e:
                self.logg.error(f"Laser Error: {e}")
        _h = {key: True for key in lasers.keys()}
        return lasers, _h

    def close(self):
        self.laser_off("all")
        for key, _l in self._h.items():
            if _l:
                self.lasers[key].disconnect()
                del self.lasers[key]

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

    def laser_off(self, laser):
        if laser == "all":
            for key, _l in self.lasers.items():
                _l.send_cmd('l0')
        else:
            for ind, ln in enumerate(laser):
                if self._h.get(ln, False):
                    self.lasers[ln].send_cmd('l0')

    def laser_on(self, laser):
        if laser == "all":
            for key, _l in self.lasers.items():
                _l.send_cmd('l1')
        else:
            for ind, ln in enumerate(laser):
                if self._h.get(ln, False):
                    self.lasers[ln].send_cmd('l1')

    def set_constant_power(self, laser, power):
        for ind, ln in enumerate(laser):
            if self._h.get(ln, False):
                self.lasers[ln].constant_power(power[ind])

    def set_constant_current(self, laser, current):
        for ind, ln in enumerate(laser):
            if self._h.get(ln, False):
                self.lasers[ln].constant_current(current[ind])

    def set_modulation_mode(self, laser, pw):
        for ind, ln in enumerate(laser):
            if self._h.get(ln, False):
                self.lasers[ln].modulation_mode(pw[ind])
                self.lasers[ln].digital_modulation(enable=1)
