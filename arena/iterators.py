import numpy
import cv2
import re
import time
from utils import *

'''
A simple data iteration wrapper for tracking sequence data, speed can be optimized.

video_list, a list file for all tracking video data, every line denotes a specific video file path with detailed image jpeg information and roi information
For example:
    video_list:
    /pathtovideo/car1.lst
    /pathtovideo/car2.lst
    ...
    car1.lst
    /pathtoimg/00001.jpeg x y width height
    ...
    Note that the roi format is the left top coordinates with the box width and height, which is consistent with the OTB100 data format.

output_size: desired size after resize
'''


class TrackingIterator(object):
    def __init__(self, video_list, output_size=(100, 100)):
        self.rng = get_numpy_rng()
        self.img_lists = []
        self.roi_lists = []
        with open(video_list) as video_fin:
            for video_line in video_fin:
                img_list, roi_list = self._read_image_list(video_line.split()[0])
                self.img_lists.append(img_list)
                self.roi_lists.append(roi_list)

        self.output_size = output_size
        self.video_num = len(self.img_lists)

    def sample(self, batch_size, length, interval_step=1):
        data_batch = numpy.zeros((batch_size, length, 3) + self.output_size)
        roi_batch = numpy.zeros((batch_size, length, 4))
        counter = 0
        while counter < batch_size:
            video_index = self.rng.randint(0, self.video_num)
            start_index = self.rng.randint(0, len(self.img_lists[video_index]) - (length - 1)*interval_step)
            for i in xrange(length):
                im = cv2.imread(self.img_lists[video_index][start_index])
                im = cv2.resize(im, self.output_size, interpolation=cv2.INTER_LINEAR)
                if im.ndim == 2:
                    im = numpy.tile(im.reshape((1, 3, 3)), (3, 1, 1))
                else:
                    im = numpy.swapaxes(im, 0, 2)
                    im = numpy.swapaxes(im, 1, 2)
                data_batch[counter, i, :, :, :] = im
            roi_batch[counter] = numpy.asarray(
                self.roi_lists[video_index][start_index:start_index+length])
            counter += 1

        return data_batch, roi_batch

    def _read_image_list(self, file_path):
        img_list = []
        roi_list = []
        with open(file_path) as img_fin:
            for img_line in img_fin:
                l = re.split('[ ,]', img_line)
                if len(l) < 5:
                    raise IOError
                img_list.append(l[0])
                roi_list.append([float(l[1]), float(l[2]), float(l[3]), float(l[4])])

        return img_list, roi_list

track_iter = TrackingIterator('/home/sliay/Documents/OTB100/tmp/video.lst', (100, 100))

start = time.time()
for i in xrange(20):
    data_batch, roi_batch = track_iter.sample(batch_size=32, length=10, interval_step=2)
    print data_batch.shape, roi_batch.shape

end = time.time()
print 'time per minibatch is %f' % ((end - start) / 20.0)
