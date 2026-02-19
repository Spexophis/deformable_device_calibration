# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import time
from dataclasses import dataclass

import PySpin
import numpy as np

from deformable_device_calibration import logger, run_threads


def read_writeable(node):
    return PySpin.IsReadable(node) and PySpin.IsWritable(node)


@dataclass
class CameraSettings:
    t_clean: float = 0.001
    t_readout: float = 0.002
    t_exposure: float = 100
    t_accumulate: float = 0
    t_kinetic: float = 0
    gain: int = 0
    bin_h: int = 1
    bin_v: int = 1
    _start_h: int = 0
    _end_h: int = 2048
    _start_v: int = 0
    _end_v: int = 1536
    _pixels_x: int = 2048
    _pixels_y: int = 1536
    _img_size: int = 0
    ps: float = 3.45  # micron
    buffer_size: int = 4
    acq_num: int = 0
    acq_first: int = 0
    acq_last: int = 0
    valid_index: int = 0

    def __post_init__(self):
        self._img_size = self._pixels_x * self._pixels_y

    @property
    def pixels_x(self) -> int:
        return self._pixels_x

    @pixels_x.setter
    def pixels_x(self, value: int):
        self._pixels_x = value
        self._img_size = self._pixels_x * self._pixels_y

    @property
    def pixels_y(self) -> int:
        return self._pixels_y

    @pixels_y.setter
    def pixels_y(self, value: int):
        self._pixels_y = value
        self._img_size = self._pixels_x * self._pixels_y

    @property
    def img_size(self) -> int:
        return self._img_size

    @property
    def start_h(self) -> int:
        return self._start_h

    @start_h.setter
    def start_h(self, value: int):
        self._start_h = value

    @property
    def start_v(self) -> int:
        return self._start_v

    @start_v.setter
    def start_v(self, value: int):
        self._start_v = value


class FLIRCamera:

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self._settings = CameraSettings()
        self.syst, self.cam_list = self._initialize_sdk()
        if self.syst and self.cam_list:
            self.cam = self.cam_list[0]
            self.node_map, self.node_map_stream = self._configure_camera()
            self._init_camera()
            self.data = None
            self.acq_thread = None
        else:
            raise RuntimeError("Failed to initiate FLIR camera.")

    def __del__(self):
        pass

    def __getattr__(self, item):
        if hasattr(self._settings, item):
            return getattr(self._settings, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def close(self):
        self.cam.DeInit()
        del self.cam
        self.cam_list.Clear()
        self.syst.ReleaseInstance()

    def _initialize_sdk(self):
        try:
            syst = PySpin.System.GetInstance()
            version = syst.GetLibraryVersion()
            self.logg.info('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))
            cam_list = syst.GetCameras()
            num_cameras = cam_list.GetSize()
            if num_cameras == 0:
                cam_list.Clear()
                syst.ReleaseInstance()
                self.logg.info('No FLIR camera detected!')
                return False, None
            else:
                self.logg.info('Number of FLIR cameras detected: %d' % num_cameras)
                return syst, cam_list
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)
            return False, None

    def _camera_info(self, nod_map):
        self.logg.info('*** FLIR CAMERA INFORMATION ***')
        try:
            node_device_information = PySpin.CCategoryPtr(nod_map.GetNode('DeviceInformation'))
            if PySpin.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PySpin.CValuePtr(feature)
                    self.logg.info('%s: %s' % (node_feature.GetName(),
                                               node_feature.ToString() if PySpin.IsReadable(
                                                   node_feature) else 'Node not readable'))
            else:
                self.logg.error('Device control information not readable.')
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)

    def _configure_camera(self):
        try:
            _tl_device = self.cam.GetTLDeviceNodeMap()
            self._camera_info(_tl_device)
            self.cam.Init()
            node_map = self.cam.GetNodeMap()
            node_map_stream = self.cam.GetTLStreamNodeMap()
            return node_map, node_map_stream
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)
            return False

    def _init_camera(self):
        self.auto_off()
        self.set_bit_depth()
        self.set_acquisition_mode(3)
        self.set_trigger_mode(1)

    def auto_off(self):
        """
        Turn off automatic gain, exposure
        """
        try:
            # Turn off automatic gain
            node_gain_auto = PySpin.CEnumerationPtr(self.node_map.GetNode('GainAuto'))
            if read_writeable(node_gain_auto):
                node_gain_auto_off = PySpin.CEnumEntryPtr(node_gain_auto.GetEntryByName('Off'))
                if PySpin.IsReadable(node_gain_auto_off):
                    node_gain_auto.SetIntValue(node_gain_auto_off.GetValue())
                    self.logg.info('Automatic gain disabled...')
                else:
                    self.logg.info('Unable to disable automatic gain (enum entry retrieval). Aborting...')
            else:
                self.logg.info('Unable to disable automatic gain (node retrieval). Aborting...')

            # Turn off automatic exposure mode
            if self.cam.ExposureAuto.GetAccessMode() == PySpin.RW:
                self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
                self.logg.info('Automatic exposure disabled...')
            else:
                self.logg.error('Unable to disable automatic exposure. Aborting...')
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)

    def set_bit_depth(self):
        try:
            if self.cam.PixelFormat.GetAccessMode() == PySpin.RW:
                self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
                self.logg.info('Pixel format set to %s...' % self.cam.PixelFormat.GetCurrentEntry().GetSymbolic())
            else:
                self.logg.error('Pixel format not available...')
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)

    def set_acquisition_mode(self, ind):
        """
        1 - Single Frame
        2 - Multi Frame
        3 - Continuous
        """
        try:
            if self.cam.AcquisitionMode.GetAccessMode() == PySpin.RW:
                if ind == 1:
                    md = PySpin.AcquisitionMode_SingleFrame
                    self.cam.AcquisitionMode.SetValue(md)
                    self.logg.info('Acquisition mode set to Single Frame...')
                elif ind == 2:
                    md = PySpin.AcquisitionMode_MultiFrame
                    self.cam.AcquisitionMode.SetValue(md)
                    self.logg.info('Acquisition mode set to Multi Frame...')
                elif ind == 3:
                    md = PySpin.AcquisitionMode_Continuous
                    self.cam.AcquisitionMode.SetValue(md)
                    self.logg.info('Acquisition mode set to Continuous...')
                else:
                    self.logg.error('Invalid acquisition mode. Aborting...')
            else:
                self.logg.error('Unable to set acquisition mode. Aborting...')
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)

    def set_roi(self):
        """
        Configures a number of settings on the camera including offsets  X and Y, width, height, and pixel format.
        These settings must be applied before BeginAcquisition() is called; otherwise, they will be read only.
        Also, it is important to note that settings are applied immediately.
        This means if you plan to reduce the width and move the x offset accordingly,
        you need to apply such changes in the appropriate order.
        """
        try:
            # Set width
            if self.cam.Width.GetAccessMode() == PySpin.RW and self.cam.Width.GetInc() != 0 and self.cam.Width.GetMax != 0:
                width_to_set = min(self.cam.Width.GetMax(), self.pixels_x)
                self.cam.Width.SetValue(width_to_set)
                self.logg.info('Width set to %i...' % self.cam.Width.GetValue())
            else:
                self.logg.error('Width not available...')

            # Set height
            if self.cam.Height.GetAccessMode() == PySpin.RW and self.cam.Height.GetInc() != 0 and self.cam.Height.GetMax != 0:
                height_to_set = min(self.cam.Height.GetMax(), self.pixels_y)
                self.cam.Height.SetValue(height_to_set)
                self.logg.info('Height set to %i...' % self.cam.Height.GetValue())
            else:
                self.logg.error('Height not available...')

            # Apply offset X
            if self.cam.OffsetX.GetAccessMode() == PySpin.RW:
                offset_x_to_set = max(self.cam.OffsetX.GetMin(), self.start_h)
                self.cam.OffsetX.SetValue(offset_x_to_set)
                self.logg.info('Offset X set to %d...' % self.cam.OffsetX.GetValue())
            else:
                self.logg.error('Offset X not available...')

            # Apply offset Y
            if self.cam.OffsetY.GetAccessMode() == PySpin.RW:
                offset_y_to_set = max(self.cam.OffsetY.GetMin(), self.start_v)
                self.cam.OffsetY.SetValue(offset_y_to_set)
                self.logg.info('Offset Y set to %d...' % self.cam.OffsetY.GetValue())
            else:
                self.logg.error('Offset Y not available...')
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)

    def set_trigger_mode(self, ind):
        """
        1 - SOFTWARE
        2 - HARDWARE

        This function configures the camera to use a trigger. First, trigger mode is
        set to off in order to select the trigger source. Once the trigger source
        has been selected, trigger mode is then enabled, which has the camera
        capture only a single image upon the execution of the chosen trigger.

        Note that if the application / user software triggers faster than frame time,
        the trigger may be dropped / skipped by the camera.
        If several frames are needed per trigger, a more reliable alternative for such case,
        is to use the multi-frame mode.
        """
        try:
            # Set TriggerSelector to FrameStart
            # This is the default for most cameras.
            if self.cam.TriggerSelector.GetAccessMode() == PySpin.RW:
                self.cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
                self.logg.info('Trigger selector set to frame start...')
            else:
                self.logg.error('Unable to get trigger selector (node retrieval). Aborting...')
            # Ensure trigger mode off
            # The trigger must be disabled in order to configure whether the source is software or hardware.
            if self.cam.TriggerMode.GetAccessMode() == PySpin.RW:
                self.cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
                self.logg.info('Trigger mode disabled...')
                if ind == 1:
                    # Select trigger source
                    # The trigger source must be set to hardware or software while trigger mode is off.
                    if self.cam.TriggerSource.GetAccessMode() == PySpin.RW:
                        self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
                        self.logg.info('Trigger source set to software...')
                    else:
                        self.logg.error('Unable to get trigger source (node retrieval). Aborting...')
                elif ind == 2:
                    if self.cam.TriggerSource.GetAccessMode() == PySpin.RW:
                        self.cam.TriggerSource.SetValue(PySpin.TriggerSource_Line3)
                        self.logg.info('Trigger source set to hardware...')
                    else:
                        self.logg.error('Unable to get trigger source (node retrieval). Aborting...')
                else:
                    self.logg.error('Invalid trigger source. Aborting...')
            else:
                self.logg.error('Unable to disable trigger mode (node retrieval). Aborting...')
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)

    def open_trigger(self):
        """
        This function turn trigger mode on in order to retrieve images using the trigger.
        """
        try:
            trigger_mode = PySpin.CEnumerationPtr(self.node_map.GetNode('TriggerMode'))
            if read_writeable(trigger_mode):
                trigger_mode_on = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName('On'))
                if PySpin.IsReadable(trigger_mode_on):
                    trigger_mode.SetIntValue(trigger_mode_on.GetValue())
                    self.logg.info('Trigger mode turned on...\n')
                else:
                    self.logg.error('Unable to turn on trigger mode (enum entry retrieval). Non-fatal error...\n')
            else:
                self.logg.error('Unable to turn on trigger mode (node retrieval). Non-fatal error...\n')
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)

    def close_trigger(self):
        """
        This function turn trigger mode off to restore the camera to a clean state.
        """
        try:
            trigger_mode = PySpin.CEnumerationPtr(self.node_map.GetNode('TriggerMode'))
            if read_writeable(trigger_mode):
                trigger_mode_off = PySpin.CEnumEntryPtr(trigger_mode.GetEntryByName('Off'))
                if PySpin.IsReadable(trigger_mode_off):
                    trigger_mode.SetIntValue(trigger_mode_off.GetValue())
                    self.logg.info('Trigger mode turned off...\n')
                else:
                    self.logg.error('Unable to turn off trigger mode (enum entry retrieval). Non-fatal error...\n')
            else:
                self.logg.error('Unable to turn off trigger mode (node retrieval). Non-fatal error...\n')
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)

    def software_trigger(self):
        try:
            # Execute software trigger
            # Blackfly and Flea3 GEV cameras need 2 second delay after software trigger
            if self.cam.TriggerSoftware.GetAccessMode() == PySpin.WO:
                self.cam.TriggerSoftware.Execute()
                self.logg.info('Software trigger emitted')
                time.sleep(2.0)
            else:
                self.logg.error('Unable to execute trigger. Aborting...')
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)

    def set_buffer(self, ind):
        """
        1 - "NewestFirst"
        2 - "OldestFirst"
        3 - "NewestOnly"
        4 - "OldestFirstOverwrite"
        """
        buffer_modes = ["", "NewestFirst", "OldestFirst", "NewestOnly", "OldestFirstOverwrite"]
        try:
            # Retrieve Buffer Handling Mode Information
            handling_mode = PySpin.CEnumerationPtr(self.node_map_stream.GetNode('StreamBufferHandlingMode'))

            if read_writeable(handling_mode):
                handling_mode_entry = PySpin.CEnumEntryPtr(handling_mode.GetCurrentEntry())

                if PySpin.IsReadable(handling_mode_entry):
                    # Set stream buffer Count Mode to manual
                    stream_buffer_count_mode = PySpin.CEnumerationPtr(
                        self.node_map_stream.GetNode('StreamBufferCountMode'))

                    if read_writeable(stream_buffer_count_mode):

                        stream_buffer_count_mode_manual = PySpin.CEnumEntryPtr(
                            stream_buffer_count_mode.GetEntryByName('Manual'))

                        if PySpin.IsReadable(stream_buffer_count_mode_manual):

                            stream_buffer_count_mode.SetIntValue(stream_buffer_count_mode_manual.GetValue())
                            self.logg.info('Stream Buffer Count Mode set to manual...')
                            # Retrieve and modify Stream Buffer Count
                            buffer_count = PySpin.CIntegerPtr(self.node_map_stream.GetNode('StreamBufferCountManual'))

                            if read_writeable(buffer_count):
                                buffer_num = min(buffer_count.GetMax(), self.buffer_size)
                                buffer_count.SetValue(buffer_num)
                                self.logg.info('Buffer count now set to: %d' % buffer_count.GetValue())
                                handling_mode_entry = handling_mode.GetEntryByName(buffer_modes[ind])
                                handling_mode.SetIntValue(handling_mode_entry.GetValue())
                                self.logg.info(
                                    '\nBuffer Handling Mode has been set to %s' % handling_mode_entry.GetDisplayName())
                            else:
                                self.logg.error('Unable to set Buffer Count (Integer node retrieval). Aborting...\n')
                        else:
                            self.logg.error('Unable to set Buffer Count Mode entry (Entry retrieval). Aborting...\n')
                    else:
                        self.logg.error('Unable to set Buffer Count Mode (node retrieval). Aborting...\n')
                else:
                    self.logg.error('Unable to set Buffer Handling mode (Entry retrieval). Aborting...\n')
            else:
                self.logg.error('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)

    def set_gain(self):
        """
        Change gain
        """
        try:
            if self.cam.Gain.GetAccessMode() == PySpin.RW:
                gain_to_set = min(self.cam.Gain.GetMax(), self.gain)
                self.cam.Gain.SetValue(gain_to_set)
                time.sleep(0.001)
                current_gain = self.cam.Gain.GetValue()
                self.logg.info(f'Gain set to {current_gain}')
            else:
                self.logg.error('Unable to set gain. Aborting...')
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)

    def set_exposure_time(self):
        """
        Set exposure time manually; exposure time recorded in microseconds
        """
        try:
            if self.cam.ExposureTime.GetAccessMode() == PySpin.RW:
                exposure_time_to_set = min(self.cam.ExposureTime.GetMax(), self.t_exposure)
                self.cam.ExposureTime.SetValue(exposure_time_to_set)
                time.sleep(0.001)
                current_exposure_time = self.cam.ExposureTime.GetValue()
                self.logg.info(f'Shutter time set to {current_exposure_time} us')
            else:
                self.logg.error('Unable to set exposure time. Aborting...')
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)

    def prepare_live(self):
        self.set_gain()
        self.set_roi()
        self.set_buffer(self.buffer_size)
        self.set_exposure_time()
        # self.open_trigger()
        self.data = run_threads.CameraDataList(self.buffer_size)
        self.acq_thread = run_threads.CameraAcquisitionThread(self, interval=0.001)

    def start_live(self):
        self.acq_thread.start()
        self.cam.BeginAcquisition()
        self.logg.info('Acquiring images...')

    def stop_live(self):
        self.acq_thread.stop()
        self.acq_thread = None
        self.cam.EndAcquisition()
        # self.close_trigger()

    def get_images(self):
        try:
            #  Retrieve next received image
            image_result = self.cam.GetNextImage(500)
            #  Ensure image completion
            if image_result.IsIncomplete():
                self.logg.info('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                return None
            else:
                image_id = image_result.GetFrameID()
                image_data = image_result.GetNDArray()
            #  Release image
            #  *** NOTES ***
            #  Images retrieved directly from the camera (i.e. non-converted images) need to be released
            #  in order to keep from filling the buffer.
            image_result.Release()
            self.data.add_element([image_data], [image_id, image_id])
        except PySpin.SpinnakerException as ex:
            self.logg.error('Error: %s' % ex)

    def get_last_image(self):
        if self.data is not None:
            return self.data.get_last_element(True)
        else:
            return None

    def get_buffered_images(self):
        if self.data is not None:
            return self.data.get_elements()
        else:
            return None

    def prepare_data_acquisition(self):
        self.set_gain()
        self.buffer_size = self.acq_num
        self.set_buffer(self.buffer_size)
        self.open_trigger()

    def start_data_acquisition(self):
        self.data = np.zeros((self.acq_num, self.pixels_x, self.pixels_y))
        self.cam.BeginAcquisition()
        self.logg.info('Acquiring images...')

    def stop_data_acquisition(self):
        self.cam.EndAcquisition()
        self.close_trigger()

    def get_data(self):
        re = self.get_image(True)
        if re is not None:
            self.data[re[1]] = re[0]
