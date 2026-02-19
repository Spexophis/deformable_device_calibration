# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import os
import time

import numpy as np
import pandas as pd
import tifffile as tf
from PyQt6.QtCore import QObject, pyqtSlot, pyqtSignal, QTimer, Qt

from . import run_threads, logger
from .utilities import image_processor as ipr


class CommandExecutor(QObject):
    sig_plt = pyqtSignal(list, list)
    psv = pyqtSignal(str)
    zsv = pyqtSignal(list, object)

    def __init__(self, dev, cwd, cmp, path, config, logg=None, cf=None):
        super().__init__()
        self.devs = dev
        self.vw = cwd
        self.ctrl_panel = self.vw.ctrl_panel
        self.viewer = self.vw.viewer
        self.ao_panel = self.vw.ao_panel
        self.intwfr = cmp.intwfr
        self.shwfr = cmp.shwfr
        self.path = path
        self.config = config
        self.cfd = cf
        self.logg = logg or logger.setup_logging()
        self._set_signal_executions()
        self._initial_setup()
        self.lasers = []
        self.task_worker = None

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _set_signal_executions(self):
        # Lasers
        self.ctrl_panel.Signal_set_laser.connect(self.set_laser)
        # Deformable Mirror
        self.ao_panel.Signal_push_actuator.connect(self.push_actuator)
        self.ao_panel.Signal_set_zernike.connect(self.set_zernike)
        self.ao_panel.Signal_set_dm.connect(self.set_dm_current)
        self.ao_panel.Signal_set_dm_flat.connect(self.set_dm_flat)
        self.ao_panel.Signal_update_cmd.connect(self.update_dm)
        self.ao_panel.Signal_save_dm.connect(self.save_dm)
        self.ao_panel.Signal_influence_function.connect(self.run_influence_function)
        # WFS
        self.ao_panel.Signal_img_shwfs_base.connect(self.set_reference_wf)
        self.ao_panel.Signal_img_wfs.connect(self.wfs)
        self.ao_panel.Signal_img_wfr_run.connect(self.run_img_wfr)
        self.ao_panel.Signal_img_shwfs_compute_wf.connect(self.run_wf_decomposition)
        self.zsv.connect(self.save_zernike_coef)
        self.ao_panel.Signal_img_shwfs_save_wf.connect(self.save_img_wf)
        # AO
        self.ao_panel.Signal_sensorlessAO_run.connect(self.run_sensorless_iteration)
        self.ao_panel.Signal_img_shwfs_correct_wf.connect(self.run_close_loop_iteration)
        self.ao_panel.Signal_sensorAO_run.connect(self.run_wfs_iteration)
        self.sig_plt.connect(self.plot_curve)

    def _initial_setup(self):
        try:
            self.laser_lists = list(self.devs.laser.lasers.keys())
            self.ao_panel.QComboBox_dms.addItem(self.devs.dfm.dm_model)
            for i in range(len(self.devs.dfm.dm_cmd)):
                self.ao_panel.QComboBox_cmd.addItem(f"{i}")
            self.ao_panel.QComboBox_cmd.setCurrentIndex(self.devs.dfm.current_cmd)
            self.logg.info("Finish setting up controllers")
        except Exception as e:
            self.logg.error(f"Initial setup Error: {e}")

    @pyqtSlot(list, bool, float)
    def set_laser(self, laser: list, sw: bool, pw: float):
        if sw:
            try:
                self.devs.laser.set_constant_power(laser, [pw])
                self.devs.laser.laser_on(laser)
            except Exception as e:
                self.logg.error(f"Cobolt Laser Error: {e}")
        else:
            try:
                self.devs.laser.laser_off(laser)
            except Exception as e:
                self.logg.error(f"Cobolt Laser Error: {e}")

    def set_lasers(self, lasers):
        pws = self.ctrl_panel.get_cobolt_laser_power("all")
        ln = []
        pw = []
        for ls in lasers:
            ln.append(self.laser_lists[ls])
            pw.append(pws[ls])
        try:
            self.devs.laser.set_modulation_mode(ln, pw)
            self.devs.laser.laser_on(ln)
        except Exception as e:
            self.logg.error(f"Cobolt Laser Error: {e}")

    def lasers_off(self):
        try:
            self.devs.laser.laser_off("all")
        except Exception as e:
            self.logg.error(f"Cobolt Laser Error: {e}")

    def set_camera_roi(self):
        try:
            expo = self.ao_panel.get_cmos_exposure()
            self.devs.camera.t_exposure = expo * 1000
            gain = self.ao_panel.get_cmos_gain()
            self.devs.camera.gain = gain
            x, y, nx, ny, bn = self.ao_panel.get_cmos_roi()
            self.devs.camera.pixels_x = nx
            self.devs.camera.start_h = x
            self.devs.camera.pixels_y = ny
            self.devs.camera.start_v = y
            self.devs.camera.bin_h = bn
            self.devs.camera.bin_v = bn
        except Exception as e:
            self.logg.error(f"Camera Error  : {e}")

    @pyqtSlot(list, list)
    def plot_curve(self, xx: list, yy: list):
        try:
            self.viewer.plot_trace(y=yy, x=xx, overlay=False)
        except Exception as e:
            self.logg.error(f"Error plotting: {e}")

    def run_task(self, task, iteration=1, parent=None):
        if getattr(self, "task_worker", None) is not None and self.task_worker.isRunning():
            return
        self.task_worker = run_threads.TaskWorker(task=task, n=iteration, parent=parent)
        self.task_worker.finished.connect(self.task_finish)
        self.task_worker.start()

    @pyqtSlot()
    def task_finish(self):
        w = self.task_worker
        self.task_worker = None
        w.deleteLater()
        self.vw.dialog.close()

    @pyqtSlot(int, float)
    def push_actuator(self, n: int, a: float):
        try:
            values = [0.] * self.devs.dfm.n_actuator
            values[n] = a
            self.devs.dfm.set_dm(self.devs.dfm.cmd_add(values, self.devs.dfm.dm_cmd[self.devs.dfm.current_cmd]))
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    @pyqtSlot(str, int, float)
    def set_zernike(self, md: str, iz: int, amp: float, factory=False):
        try:
            if factory:
                self.devs.dfm.set_dm(
                    self.devs.dfm.cmd_add([i * amp for i in self.devs.dfm.z2c[iz]],
                                          self.devs.dfm.dm_cmd[self.devs.dfm.current_cmd]))
            else:
                self.devs.dfm.temp_cmd.append(
                    self.devs.dfm.cmd_add(self.devs.dfm.get_zernike_cmd(iz, amp, md),
                                          self.devs.dfm.dm_cmd[self.devs.dfm.current_cmd]))
                self.devs.dfm.set_dm(self.devs.dfm.temp_cmd[-1])
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    @pyqtSlot(int)
    def set_dm_current(self, i: int):
        try:
            self.devs.dfm.set_dm(self.devs.dfm.dm_cmd[i])
            self.devs.dfm.current_cmd = i
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    @pyqtSlot()
    def set_dm_flat(self):
        if int(self.ao_panel.get_cmd_index()) == self.devs.dfm.current_cmd:
            self.devs.dfm.write_flat_cmd(t=time.strftime("%Y_%m_%d_%H_%M"),
                                         cmd=self.devs.dfm.dm_cmd[self.devs.dfm.current_cmd])

    @pyqtSlot()
    def update_dm(self):
        try:
            self.devs.dfm.dm_cmd.append(self.devs.dfm.temp_cmd[-1])
            self.ao_panel.update_cmd_index()
            self.devs.dfm.set_dm(self.devs.dfm.dm_cmd[-1])
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    @pyqtSlot()
    def save_dm(self):
        try:
            t = time.strftime("%Y%m%d_%H%M%S_")
            self.devs.dfm.write_cmd(self.path, t, flatfile=False)
            self.logg.info('DM cmd saved')
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    def set_img_wfs(self, md):
        if md == "Interferometry":
            self.set_int_wfs()
        elif md == "ShackHartmann":
            self.set_sh_wfs()
        else:
            self.logg.error(f"Unknown Influence Function Type: {md}")
            
    def set_int_wfs(self):
        parameters = self.ao_panel.get_parameters_int()
        self.intwfr.update_parameters(parameters)
        self.logg.info('Interferometry parameter updated')

    def set_sh_wfs(self):
        parameters = self.ao_panel.get_parameters_foc()
        self.shwfr.pixel_size = 3.45 / 1000
        self.shwfr.update_parameters(parameters)
        self.logg.info('SHWFS parameter updated')

    def prepare_wfs(self, md):
        self.set_camera_roi()
        self.devs.camera.prepare_live()
        self.viewer.switch_camera(self.devs.camera.pixels_y, self.devs.camera.pixels_x)
        self.set_img_wfs(md)

    @pyqtSlot(bool, str)
    def wfs(self, sw: bool, md: str):
        if sw:
            try:
                self.prepare_wfs(md)
                self.logg.info(f"Finish preparing wfs")
            except Exception as e:
                self.logg.error(f"Error preparing wfs: {e}")
                return
            self.start_wfs()
        else:
            self.stop_wfs()
            if self.viewer.wfr_mode:
                self.run_img_wfr(False)

    def start_wfs(self):
        try:
            self.devs.camera.start_live()
            self.devs.camera.data.on_update(self.viewer.on_camera_update_from_thread)
            self.logg.info("WFS Started")
        except Exception as e:
            self.logg.error(f"Error starting wfs: {e}")
            self.stop_video()
            return

    def stop_wfs(self):
        try:
            self.devs.camera.stop_live()
            self.logg.info(r"WFS Stopped")
        except Exception as e:
            self.logg.error(f"Error stopping wfs: {e}")

    @pyqtSlot()
    def set_reference_wf(self):
        try:
            self.shwfr.ref = self.devs.camera.get_last_image()
            self.logg.info('shwfs base set')
        except Exception as e:
            self.logg.error(f"Error setting shwfs base: {e}")

    @pyqtSlot(bool, str)
    def run_img_wfr(self, on: bool, md: str):
        if on:
            if md == "Interferometry":
                wfr = self.intwfr
            elif md == "ShackHartmann":
                wfr = self.shwfr
                self.shwfr.method = self.ao_panel.get_gradient_method_img()
            else:
                wfr = None
                self.logg.error(f"Unknown Influence Function Type: {md}")
            if getattr(self.viewer, "wfr_worker", None) is None and wfr is not None:
                self.viewer.wfr_worker = run_threads.WFRWorker(fps=8, op=wfr, parent=self.viewer)
                self.viewer.wfr_worker.wfr_ready.connect(self.viewer.on_wfr_frame, Qt.ConnectionType.QueuedConnection)
                self.viewer.wfr_worker.wfr_ready.connect(self.show_wf_metric)
                self.viewer.wfr_worker.start()
                self.viewer.wfr_mode = True
        else:
            self._cleanup_psr_worker()
            self.viewer.wfr_mode = False

    def _cleanup_wfr_worker(self):
        """Properly cleanup wfr worker"""
        worker = getattr(self.viewer, "wfr_worker", None)
        if worker is not None:
            # Stop the worker thread
            worker.stop()

            # Disconnect all signals
            try:
                worker.wfr_ready.disconnect()
            except TypeError:
                pass  # Already disconnected

            # Clear data references
            worker.clear_data()

            # Delete worker
            worker.deleteLater()  # Qt will delete when safe
            self.viewer.wfr_worker = None

    def show_wf_metric(self, wf_img):
        try:
            self.ao_panel.display_img_wf_properties(ipr.img_statistics(wf_img))
        except Exception as e:
            self.logg.error(f"SHWFS Wavefront Show Error: {e}")

    @pyqtSlot(bool)
    def run_wf_decomposition(self, on: bool):
        if on:
            self.viewer.wfr_decomp = True
        else:
            self.viewer.wfr_decomp = False

    @pyqtSlot(list, object)
    def save_zernike_coef(self, zdx: list, za: object):
        df = pd.DataFrame({'mods': zdx, 'amps': za})
        fn = self.vw.get_file_dialog()
        if fn is not None:
            file_path = fn + '_' + time.strftime("%Y%m%d%H%M%S")
        else:
            file_path = os.path.join(self.path, time.strftime("%Y%m%d%H%M%S"))
        df.to_excel(file_path + '_zernike_coefficients.xlsx', index=False)

    @pyqtSlot()
    def save_img_wf(self):
        fn = self.vw.get_file_dialog()
        if fn is not None:
            file_name = os.path.join(self.path, time.strftime("%Y%m%d%H%M%S") + "_" + fn)
        else:
            file_name = os.path.join(self.path, time.strftime("%Y%m%d%H%M%S"))
        self.shwfr.save_wfs_results(file_name, self.devs.dfm)

    def close_loop_correction(self, n, md, fnd):
        data = [self.shwfr.ref]
        for i in range(n):
            self.shwfr.meas = self.devs.camera.get_last_image()
            data.append(self.shwfr.meas)
            gdx, gdy = self.shwfr.get_gradient_xy()
            self.devs.dfm.get_correction((gdx, gdy), md)
            self.devs.dfm.set_dm(self.devs.dfm.dm_cmd[-1])
            self.ao_panel.update_cmd_index()
            i = int(self.ao_panel.get_cmd_index())
            self.devs.dfm.current_cmd = i
            self.logg.info(f"Successful close loop correction")
        tf.imwrite(str(fnd), np.array(data))

    def close_loop_iteration(self):
        try:
            self.prepare_wfs()
        except Exception as e:
            self.logg.error(f"Error preparing close loop iteration: {e}")
            return
        try:
            md = self.ao_panel.get_img_wfs_method()
            name = time.strftime("%Y%m%d_%H%M%S_") + self.devs.dfm.dm_serial + '_close_loop_iterations_' + md + r".tif"
            fnd = os.path.join(self.path, name)
            self.devs.camera.start_live()
            n = self.ao_panel.QSpinBox_close_loop_number.value()
            time.sleep(2)
            self.close_loop_correction(n, md, fnd)
        except Exception as e:
            self.logg.error(f"Error running close loop iteration: {e}")
            self.stop_wfs()
            return
        self.stop_wfs()

    @pyqtSlot()
    def run_close_loop_iteration(self):
        self.vw.get_dialog(txt="Close Loop Iteration")
        self.run_task(lambda: self.close_loop_iteration())

    def sensor_iteration(self, zn, dms):
        ims = []
        zns = []
        for dmsp in dms:
            self.devs.dfm.set_dm(dmsp)
            time.sleep(0.064)
            img = self.devs.camera.get_last_image()
            ims.append(img)
            self.shwfr.meas = img
            gdx, gdy = self.shwfr.get_gradient_xy()
            temp = ipr.get_eigen_coefficients(np.concatenate((gdx.flatten(), gdy.flatten())), self.devs.dfm.zslopes, 14)
            zns.append(np.abs(temp[zn]))
        return ims, zns

    def wfs_iteration(self):
        try:
            self.prepare_wfs()
        except Exception as e:
            self.logg.error(f"Error preparing wfs iteration: {e}")
            return
        try:
            md = self.ao_panel.get_img_wfs_method()
            name = time.strftime("%Y%m%d_%H%M%S_") + self.devs.dfm.dm_serial + '_wfs_iterations_' + md + r".tif"
            new_folder = os.path.join(self.path, name)
            os.makedirs(new_folder, exist_ok=True)
            self.logg.info(f'Directory {new_folder} has been created successfully.')
        except Exception as e:
            self.logg.error(f'Error creating directory for wfs iteration: {e}')
            return
        try:
            self.devs.camera.start_live()
            time.sleep(2)
            cmd = self.devs.dfm.dm_cmd[self.devs.dfm.current_cmd]
            self.devs.dfm.set_dm(cmd)
            mode_start, mode_stop, _, amp_step, amp_step_number = self.ao_panel.get_sensorless_iteration()
            md = self.ao_panel.get_img_wfs_method()
            self.shwfr.meas = self.devs.camera.get_last_image()
            data = [self.shwfr.ref, self.shwfr.meas]
            gdx, gdy = self.shwfr.get_gradient_xy()
            zcs = ipr.get_eigen_coefficients(np.concatenate((gdx.flatten(), gdy.flatten())), self.devs.dfm.zslopes, 14)
            self.sig_plt.emit(np.arange(zcs.size), zcs)
            amp_starts = [- zc - amp_step * int(amp_step_number / 2) for zc in zcs]
            results = [('Mode', 'Amp', 'Metric')]
            za = []
            mv = []
            zp = [0] * self.devs.dfm.n_zernike
            for mode in range(mode_start, mode_stop + 1):
                self.vw.dialog_text.setText(f"Zernike mode #{mode}")
                amp_range = [amp_starts[mode] + step_number * amp_step for step_number in range(amp_step_number)]
                labels = ["zm%0.2d_amp%.4f" % (mode, amp) for amp in amp_range]
                cmds = [self.devs.dfm.cmd_add(self.devs.dfm.get_zernike_cmd(mode, amp, method=md), cmd) for amp in amp_range]
                images, zma = self.sensor_iteration(mode, cmds)
                self.sig_plt.emit(amp_range, zma)
                pm = ipr.valley_find(amp_range, zma)
                if isinstance(pm, str):
                    self.logg.error(f"zernike mode #{mode} " + pm)
                else:
                    zp[mode] = pm
                    cmd = self.devs.dfm.cmd_add(self.devs.dfm.get_zernike_cmd(mode, pm, method=md), cmd)
                    self.devs.dfm.set_dm(cmd)
                    self.logg.info("set mode %d at value of %.4f" % (mode, pm))
                for amp, mt in zip(amp_range, zma):
                    results.append((mode, amp, mt))
                za.extend(amp_range)
                mv.extend(zma)
                fn = os.path.join(str(new_folder), f"zernike mode #{mode}.tiff")
                with tf.TiffWriter(fn) as tif:
                    for img, label in zip(images, labels):
                        tif.write(img, description=label)
            self.devs.dfm.set_dm(cmd)
            time.sleep(0.064)
            fmg = self.devs.camera.get_last_image()
            data.append(fmg)
            fn = new_folder + r"\shwfs.tiff"
            tf.imwrite(str(fn), np.array(data))
            self.devs.dfm.dm_cmd.append(cmd)
            self.ao_panel.update_cmd_index()
            i = int(self.ao_panel.get_cmd_index())
            self.devs.dfm.current_cmd = i
            self.devs.dfm.write_cmd(new_folder, '_')
            self.devs.dfm.save_sensorless_results(os.path.join(str(new_folder), 'results.xlsx'), za, mv, zp)
        except Exception as e:
            self.logg.error(f"Error running close loop iteration: {e}")
            self.stop_wfs()
            return
        self.stop_wfs()

    @pyqtSlot(str)
    def run_wfs_iteration(self, md):
        self.vw.get_dialog(txt="WFS Iteration")
        self.run_task(lambda: self.wfs_iteration(md))

    def influence_function(self, md):
        if md == "Interferometry":
            self.influence_function_int()
        elif md == "ShackHartmann":
            self.influence_function_sh()
        else:
            self.logg.error(f"Unknown Influence Function Type: {md}")

    def influence_function_int(self):
        try:
            self.prepare_wfs("Interferometry")
        except Exception as e:
            self.logg.error(f"Error preparing influence function: {e}")
            return
        try:
            fd = os.path.join(self.path, time.strftime("%Y%m%d%H%M") + '_influence_function')
            os.makedirs(fd, exist_ok=True)
            self.logg.info(f'Directory {fd} has been created successfully.')
        except Exception as er:
            self.logg.error(f'Error creating influence function directory: {er}')
            return
        try:
            amps = np.arange(-0.2, 0.201, 0.01)
            self.devs.camera.start_live()
            time.sleep(0.032)
            for i in range(self.devs.dfm.n_actuator):

                shimg = []
                self.vw.dialog_text.setText(f"actuator {i}")
                values = [0.] * self.devs.dfm.n_actuator
                self.devs.dfm.set_dm(values)
                time.sleep(0.4)
                temp = self.devs.camera.get_buffered_images()
                temp = np.average(temp, axis=0)
                shimg.append(temp)

                for a in amps:

                    values[i] = a
                    self.devs.dfm.set_dm(values)
                    time.sleep(0.4)
                    temp = self.devs.camera.get_buffered_images()
                    temp = np.average(temp, axis=0)
                    shimg.append(temp)

                tf.imwrite(fd + r'/' + 'actuator_' + str(i) + '_step_' + str(0.01) + '_range_' + str(2) + '.tif', np.asarray(shimg))
        except Exception as e:
            self.logg.error(f"Error running influence function: {e}")
            self.stop_wfs()
            return
        try:
            self.vw.dialog_text.setText(f"computing influence function")
            dmn = self.ao_panel.QComboBox_dms.currentText()
            self.shwfr.generate_influence_matrices(amps, data_folder=fd, dm=self.devs.dfm, sv=self.config, cfd=self.cfd)
        except Exception as e:
            self.logg.error(f"Error computing influence function: {e}")
            self.stop_wfs()
            return
        self.stop_wfs()

    def influence_function_sh(self):
        try:
            self.prepare_wfs("ShackHartmann")
        except Exception as e:
            self.logg.error(f"Error preparing influence function: {e}")
            return
        try:
            fd = os.path.join(self.path, time.strftime("%Y%m%d%H%M") + '_influence_function')
            os.makedirs(fd, exist_ok=True)
            self.logg.info(f'Directory {fd} has been created successfully.')
        except Exception as er:
            self.logg.error(f'Error creating influence function directory: {er}')
            return
        try:
            amps = np.arange(-0.2, 0.201, 0.01)
            self.devs.camera.start_live()
            time.sleep(0.032)
            for i in range(self.devs.dfm.n_actuator):

                shimg = []
                self.vw.dialog_text.setText(f"actuator {i}")
                values = [0.] * self.devs.dfm.n_actuator
                self.devs.dfm.set_dm(values)
                time.sleep(0.4)
                temp = self.devs.camera.get_buffered_images()
                temp = np.average(temp, axis=0)
                shimg.append(temp)

                for a in amps:

                    values[i] = a
                    self.devs.dfm.set_dm(values)
                    time.sleep(0.4)
                    temp = self.devs.camera.get_buffered_images()
                    temp = np.average(temp, axis=0)
                    shimg.append(temp)

                tf.imwrite(fd + r'/' + 'actuator_' + str(i) + '_step_' + str(0.01) + '_range_' + str(2) + '.tif', np.asarray(shimg))
        except Exception as e:
            self.logg.error(f"Error running influence function: {e}")
            self.stop_wfs()
            return
        try:
            self.vw.dialog_text.setText(f"computing influence function")
            dmn = self.ao_panel.QComboBox_dms.currentText()
            self.shwfr.generate_influence_matrices(amps, data_folder=fd, dm=self.devs.dfm, sv=self.config, cfd=self.cfd)
        except Exception as e:
            self.logg.error(f"Error computing influence function: {e}")
            self.stop_wfs()
            return
        self.stop_wfs()

    @pyqtSlot(str)
    def run_influence_function(self, md):
        self.vw.get_dialog(txt="Influence Function")
        self.run_task(lambda: self.influence_function(md))
