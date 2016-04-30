import numpy
import cv2
import re
import time
from utils import *
import matplotlib.pyplot as plt
import mxnet.ndarray as nd

'''
A simple data iteration wrapper for tracking sequence data, speed can be optimized.

video_list, a list file for all tracking video data, every line denotes a specific video file path with detailed image jpeg information and roi information
For example:
    video_list:
    /pathtovideo/car1.lst
    /pathtovideo/car2.lst
    ...
    car1.lst
    /pathtoimg/00001.jpeg x y sx sy
    ...
    Note that the roi format is the left top coordinates with the box width and height, which is consistent with the OTB100 data format.

output_size: desired size after resize
'''


class TrackingIterator(object):
    def __init__(self, video_list, output_size=(100, 100), resize=True, ctx=get_default_ctx()):
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
        self.resize = resize
        self.ctx = ctx

    '''
    Get the image mean, shape should be (batch, timestep, channel, height, width)
    '''
    def img_mean(self, shape):
        nd_mean = nd.array([123.68, 116.779, 103.939], ctx=self.ctx)
        nd_mean = nd_mean.reshape((1, 3, 1, 1))
        nd_mean = nd_mean.broadcast_to(shape)
        return nd_mean

    '''
    Choose a video and draw minibatch samples from this video.

    '''
    def sample(self, length=20, batch_size=1, interval_step=1):
        assert 1 == batch_size
        video_index = self.rng.randint(0, self.video_num)
        # make sure choose video have enough frames
        while len(self.img_lists[video_index]) < length * interval_step:
            video_index = self.rng.randint(0, self.video_num)
        print 'sampled image from video %s\n' % self.img_lists[video_index][0]
        im_shape = cv2.imread(self.img_lists[video_index][0]).shape
        if self.resize:
            seq_data_batch = numpy.zeros((length, 3) + self.output_size, dtype=numpy.uint8)
        else:
            seq_data_batch = numpy.zeros((length, 3) + (im_shape[0], im_shape[1]),
                                         dtype=numpy.uint8)
        seq_roi_batch = numpy.zeros((length, 4), dtype=numpy.float32)
        counter = 0
        while counter < batch_size:
            start_index = self.rng.randint(0, len(self.img_lists[video_index]) - (length - 1)*interval_step)
            for i in xrange(length):
                im = cv2.imread(self.img_lists[video_index][start_index+i])
                if self.resize:
                    im = cv2.resize(im, (self.output_size[1], self.output_size[0]),
                                interpolation=cv2.INTER_LINEAR)
                if im.ndim == 2:
                    im = numpy.tile(im.reshape((1, 3, 3)), (3, 1, 1))
                else:
                    im = numpy.rollaxis(im, 2)
                seq_data_batch[i, :, :, :] = im[::-1, :, :]
            seq_roi_batch[:] = numpy.asarray(
                self.roi_lists[video_index][start_index:start_index+length], dtype=numpy.float32)
            counter += 1
        seq_roi_batch[:, 0:2] += seq_roi_batch[:, 2:4] / 2 - 1
        seq_roi_batch[:, ::2] = seq_roi_batch[:, ::2] / im_shape[1]
        seq_roi_batch[:, 1::2] = seq_roi_batch[:, 1::2] / im_shape[0]
        seq_data_batch = nd.array(seq_data_batch, ctx=self.ctx)
        seq_data_batch -= self.img_mean(seq_data_batch.shape)
        seq_roi_batch = nd.array(seq_roi_batch, ctx=self.ctx)
        return seq_data_batch, seq_roi_batch

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

'''
Simple test for the performance of the tracking iterator.
'''
if __name__ == '__main__':
    track_iter = TrackingIterator('D:\\HKUST\\2-2\\learning-to-track\\datasets\\OTB100-processed\\otb100-video.lst',
                                  output_size=(240, 320), resize=False)

    start = time.time()
    for i in xrange(20):
        data_batch, roi_batch = track_iter.sample(batch_size=1, length=20, interval_step=2)
        print data_batch.shape, roi_batch.shape
        # for k in xrange(10):
        #     print roi_batch[0, k]
        #     cur_img = data_batch[0, k]
        #     cur_img = numpy.swapaxes(cur_img, 0, 2)
        #     cur_img = numpy.swapaxes(cur_img, 1, 0)
        #     print cur_img.shape
        #     plt.imshow(cur_img)
        #     plt.show()
        # print roi_batch[0, :, :]

    end = time.time()
    print 'time per minibatch is %f' % ((end - start) / 20.0)
