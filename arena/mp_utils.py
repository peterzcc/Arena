from enum import Enum
import numpy as np
import multiprocessing as mp
from multiprocessing import sharedctypes as mpshared
import ctypes
import logging


class ProcessState(Enum):
    terminate = 0
    start = 1
    stop = 2


class RenderOption(Enum):
    off = 0
    on = 1
    one_episode = 2

def force_map(func, x):
    return list(map(func, x))


def create_shared_memory(src):
    dtype = ctypes.c_float
    if isinstance(src, np.ndarray):
        srctype = src.dtype
        if srctype == np.float32:
            dtype = ctypes.c_float
        elif srctype == np.float64 or srctype == float:
            dtype = ctypes.c_double
        elif srctype == np.uint8:
            dtype = ctypes.c_uint8
        elif srctype == bool:
            dtype = ctypes.c_uint8
        else:
            raise ValueError("Unsupported arg")
        shape = src.shape

        return mpshared.RawArray(dtype, src.ravel()), shape, srctype
    elif isinstance(src, bool):
        dtype = ctypes.c_bool
        srctype = bool
    elif isinstance(src, int):
        dtype = ctypes.c_int64
        srctype = int
    elif isinstance(src, float):
        dtype = ctypes.c_double
        srctype = float
    else:
        raise ValueError("Unsupported arg")
    shape = (1,)
    var = mpshared.RawArray(dtype, [src])
    return var, shape, srctype


class FastPipe(object):
    def __init__(self, data_dict: dict):
        self.var_dict = {}
        self.shape_dict = {}
        self.dtype_dict = {}
        for k, v in data_dict.items():
            this_var, this_shape, this_dtype = create_shared_memory(v)
            self.var_dict[k] = this_var
            self.dtype_dict[k] = this_dtype
            self.shape_dict[k] = this_shape
        self.event_signal = mp.Event()
        self.event_signal.clear()
        self.npdata = None

    def prepare_np(self):
        if self.npdata is None:
            self.npdata = {}
            for k, v in self.var_dict.items():
                this_dtype = self.dtype_dict[k]
                this_shape = self.shape_dict[k]
                this_var = self.var_dict[k]
                buffer = np.frombuffer(this_var, dtype=this_dtype)
                self.npdata[k] = buffer.reshape(this_shape)

    def send(self, data):
        self.prepare_np()
        for k, v in self.var_dict.items():
            self.npdata[k][:] = data[k]
        self.event_signal.set()

    def recv(self, timeout=None):
        self.prepare_np()
        success = self.event_signal.wait(timeout=timeout)
        if success:
            self.event_signal.clear()
        return self.npdata

    def poll(self, timeout):
        return self.event_signal.wait(timeout=timeout)
