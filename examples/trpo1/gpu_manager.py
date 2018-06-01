from tensorflow.python.client import device_lib
import logging


class GpuManager(object):
    GPU_DEVICES = None
    CURRENT_DEVICE_ID = 0

    @staticmethod
    def find_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        GpuManager.GPU_DEVICES = [x.name for x in local_device_protos if x.device_type == 'GPU']

    @staticmethod
    def get_a_gpu_and_change():
        num_gpu = len(GpuManager.GPU_DEVICES)
        if num_gpu == 0:
            return '/cpu:0'
        this_gpu = GpuManager.GPU_DEVICES[GpuManager.CURRENT_DEVICE_ID]
        GpuManager.CURRENT_DEVICE_ID = (GpuManager.CURRENT_DEVICE_ID + 1) % num_gpu
        logging.info("current device: {}".format(this_gpu))
        return this_gpu
