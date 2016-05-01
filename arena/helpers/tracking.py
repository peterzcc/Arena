import re
import numpy
import os
import cv2
import mxnet as mx


def get_roi_center_size(roi):
    if type(roi) is not list:
        roi = mx.symbol.SliceChannel(roi, num_outputs=2, axis=1)
    roi_center = roi[0]
    roi_size = roi[1]
    return roi_center, roi_size


def roi_transform(anchor_roi, roi):
    anchor_center, anchor_size = get_roi_center_size(anchor_roi)
    roi_center, roi_size = get_roi_center_size(roi)
    transformed_center = (roi_center - anchor_center) / anchor_size
    transformed_size = mx.symbol.log(roi_size / anchor_size)
    transformed_center = mx.symbol.BlockGrad(transformed_center)
    transformed_size = mx.symbol.BlockGrad(transformed_size)
    return transformed_center, transformed_size


def roi_transform_inv(anchor_roi, transformed_roi):
    anchor_center, anchor_size = get_roi_center_size(anchor_roi)
    transformed_roi_center, transformed_roi_size = get_roi_center_size(transformed_roi)
    roi_center = anchor_center + transformed_roi_center * anchor_size
    roi_size = mx.symbol.exp(transformed_roi_size) * anchor_size
    roi_center = mx.symbol.clip_zero_one(roi_center)
    roi_size = mx.symbol.clip_zero_one(roi_size)
    roi_center = mx.symbol.BlockGrad(roi_center)
    roi_size = mx.symbol.BlockGrad(roi_size)
    return roi_center, roi_size


def get_timestamp(key):
    l = re.findall('_t(\d+)', key)
    assert len(l) == 1
    return int(l[0])

'''
calculate the central pixel error between two rois
im, shape (3, height, width)
roi, gt, shape (batch_size, 4), normalized version in [0, 1]
'''
def calc_CPE(im, roi, gt):
    width = im.shape[2]
    height = im.shape[1]
    roi = roi * [width, height, width, height]
    gt = gt * [width, height, width, height]
    center_roi = roi[:, 0:2]
    center_gt = gt[:, 0:2]
    cpe = numpy.sqrt(numpy.sum((center_gt - center_roi) ** 2, axis=1))
    return cpe

'''
calculate the overlap ratio between two rois
im, shape (3, height, width)
roi, gt, shape (batch_size, 4), normalized version in [0, 1]
'''
def cal_rect_int(im, roi, gt):
    width = im.shape[2]
    height = im.shape[1]

    roi = roi * [width, height, width, height]
    gt = gt * [width, height, width, height]
    left_roi = roi[:, 0] - roi[:, 2]/2
    bottom_roi = roi[:, 1] - roi[:, 3]/2
    right_roi = left_roi + roi[:, 2] - 1
    top_roi = bottom_roi + roi[:, 3] - 1

    left_gt = gt[:, 0] - gt[:, 2]/2
    bottom_gt = gt[:, 1] - gt[:, 3]/2
    right_gt = left_gt + gt[:, 2] - 1
    top_gt = bottom_gt + gt[:, 3] - 1

    tmp = numpy.maximum(0, numpy.minimum(right_gt, right_roi) - numpy.maximum(left_roi, left_gt) + 1) * numpy.maximum(0, numpy.minimum(top_roi, top_gt) - numpy.maximum(bottom_roi, bottom_gt) + 1)
    area_roi = roi[:, 2] * roi[:, 3]
    area_gt = gt[:, 2] * gt[:, 3]
    overlap = tmp / (area_gt + area_roi - tmp)
    return overlap


if __name__ == '__main__':
    rect = numpy.loadtxt(os.path.join('/home/sliay/Documents/OTB100/Basketball', 'groundtruth_rect.txt'), delimiter=',')
    im = cv2.imread('/home/sliay/Documents/OTB100/Basketball/img/0001.jpg')
    rect[:, 0:2] += rect[:, 2:4]/2 - 1

    rect[:, ::2] = rect[:, ::2] / im.shape[1]
    rect[:, 1::2] = rect[:, 1::2] / im.shape[0]
    A = rect[0:10, :]
    B = rect[10:20, :]
    overlap = cal_rect_int(im.transpose(2, 0, 1), A, B)
    print overlap
    cpe = calc_CPE(im.transpose(2, 0 ,1), A, B)
    print cpe

