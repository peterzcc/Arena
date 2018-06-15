import numpy as np
import h5py
import os
import logging
import time

class Ermemory(object):
    def __init__(self, shape, dtype=np.float32, max_size=None):
        self.shape = shape
        self.dtype = dtype
        self.max_count = max_size

    def sample(self, n):
        pass

    def get_ave_R(self):
        pass

    def prepare_sample(self, n):
        pass

    def insert(self, x, ave_R=None):
        pass


class H5Ermemory(Ermemory):
    def __init__(self, shape, path, read_only,
                 cache_size=32,
                 dtype=np.float32, max_size=None,
                 read_delay=10.):
        self.path = path
        self.data_count = 0
        self.data_top = 0
        self.read_only = read_only
        self.sample_cache = None
        self.sample_cache_top = 0
        self.has_obtained_data = False
        self.cache_size = cache_size
        self.read_delay = read_delay
        self.file_first_exist_time = None
        if not self.read_only:
            f = h5py.File(self.path, 'w', libver='latest')
            if max_size is not None:
                f.create_dataset("data", shape=(max_size, *shape), dtype=dtype)
                f["data"].attrs['n'] = self.data_count
                f["data"].attrs['ave_R'] = 0.
                f.swmr_mode = True
                logging.debug("creating dataset:{}".format(path))
            else:
                raise NotImplementedError
            self._push_count(f)
            self.f = f
        else:
            self.f = None

        super(H5Ermemory, self).__init__(shape, dtype, max_size)

    def check_file_existane_and_wait_if_first_appears(self):
        if self.has_obtained_data:
            return True
        if os.path.exists(self.path):
            if self.file_first_exist_time is None:
                self.file_first_exist_time = time.clock()
                return False
            elif time.clock() < self.file_first_exist_time + self.read_delay:
                return False
            else:
                self.has_obtained_data = True
                return True
        else:
            return False

    def sample(self, n):
        if self.sample_cache is None \
                or self.cache_size - self.sample_cache_top < n:
            if self.cache_size < n:
                self.cache_size = n
            self.prepare_sample(self.cache_size)
            if self.sample_cache is None:
                return None
        result = self.sample_cache[self.sample_cache_top:(self.sample_cache_top + n)]
        result = result[0] if n == 1 else result
        self.sample_cache_top += n
        return result

    def _pull_count(self, f):
        self.data_count = self._data_count(f)

    def _pull_data(self, slc, n, f):
        dst: h5py.Dataset = f["data"]
        data = np.empty(shape=[n, *self.shape], dtype=self.dtype)
        dst.read_direct(data, source_sel=slc)
        return data

    def _get_ave_R(self, f):
        return f["data"].attrs['ave_R']

    def _push_data(self, data, slc, n, f):
        dst: h5py.Dataset = f["data"]
        dst.write_direct(data, source_sel=np.s_[0:n], dest_sel=slc)
        return data

    def _push_count(self, f):
        f["data"].attrs['n'] = self.data_count

    def _save_ave_R(self, f, ave_R):
        f["data"].attrs['ave_R'] = ave_R

    def _data_count(self, f):
        return f["data"].attrs['n']

    def _sample_id(self, n):
        choices = np.random.choice(self.data_count, n, replace=False)
        choices.sort()
        return choices

    def get_ave_R(self):
        if self.check_file_existane_and_wait_if_first_appears():
            with h5py.File(self.path, 'r', swmr=True, libver='latest') as f:
                ave_R = self._get_ave_R(f)
                return ave_R
        else:
            return None

    def prepare_sample(self, n):
        if self.check_file_existane_and_wait_if_first_appears():
            with h5py.File(self.path, 'r', swmr=True, libver='latest') as f:
                self._pull_count(f)
                if self.data_count < n:
                    self.sample_cache = None
                    self.sample_cache_top = 0
                    return None
                sample_ids = list(self._sample_id(n))
                self.sample_cache = self._pull_data(sample_ids, n, f)
                np.random.shuffle(self.sample_cache)
                self.sample_cache_top = 0
        else:
            self.sample_cache = None
            return None

    def insert(self, x, ave_R=None):
        n = x.shape[0]
        if self.data_top + n > self.max_count:
            self.data_top = 0
        slc = np.s_[self.data_top:(self.data_top + n)]
        # with h5py.File(self.path, 'r+', swmr=True) as f:
        f = self.f
        self._push_data(x, slc, n, f)
        self.data_count = min(self.max_count, self.data_count + n)
        self._push_count(f)
        self.data_top += n

        if ave_R is not None:
            self._save_ave_R(f, ave_R)
        self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()



def main():
    shape = ()
    name = "test_dataset"
    path = "./test_data/{}.h5".format(name)
    read_only = False
    max_size = 1024
    test_mem = H5Ermemory(shape, path, read_only, cache_size=32, dtype=np.float32, max_size=max_size)
    test_mem.insert(np.arange(64))
    x = test_mem.sample(1)
    pass


if __name__ == "__main__":
    main()
