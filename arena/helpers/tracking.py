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

'''
Name: roi_transform
Description: Transform the roi based on the anchor

new_cx = (cx - anchor_cx) / anchor_sx
new_cy = (cy - anchor_cy) / anchor_sy
new_sx = log(sx/anchor_sx)
new_sy = log(sy/anchor_sy)

'''

def roi_transform(anchor_roi, roi, eps=1E-9):
    anchor_center, anchor_size = get_roi_center_size(anchor_roi)
    roi_center, roi_size = get_roi_center_size(roi)
    transformed_center = (roi_center - anchor_center) / (anchor_size + eps)
    transformed_size = mx.symbol.log((roi_size + eps) / (anchor_size + eps))
    transformed_center = mx.symbol.BlockGrad(transformed_center)
    transformed_size = mx.symbol.BlockGrad(transformed_size)
    return transformed_center, transformed_size


'''
Name: roi_transform_inv
Description: Transform back the roi based on the anchor. The result will be clipped to [0, 1] after the transformation

cx = anchor_cx + transformation_cx * anchor_sx
cy = anchor_cy + transformation_cy * anchor_sy
sx = exp(transformation_sx) * anchor_sx
sy = exp(transformation_sy) * anchor_sy

'''

def roi_transform_inv(anchor_roi, transformed_roi):
    anchor_center, anchor_size = get_roi_center_size(anchor_roi)
    transformed_roi_center, transformed_roi_size = get_roi_center_size(transformed_roi)
    roi_center = anchor_center + transformed_roi_center * anchor_size
    roi_size = mx.symbol.exp(transformed_roi_size) * anchor_size
    roi_center = mx.symbol.clip_zero_one(roi_center)
    roi_size = mx.symbol.clip_zero_one((roi_size - 0.001)/0.999) * 0.999 + 0.001
    roi_center = mx.symbol.BlockGrad(roi_center)
    roi_size = mx.symbol.BlockGrad(roi_size)
    return roi_center, roi_size


def get_timestamp(key):
    l = re.findall('_t(\d+)', key)
    assert len(l) == 1
    return int(l[0])


def get_attention_step(key):
    l = re.findall('_step(\d+)', key)
    assert len(l) == 1
    return int(l[0])

'''
calculate the central pixel error between two rois
im, shape (3, height, width)
roi, gt, shape (batch_size, 4), normalized version in [0, 1]
'''
def calc_CPE(im_height, im_width, roi, gt):
    roi = roi * [im_width, im_height, im_width, im_height]
    gt = gt * [im_width, im_height, im_width, im_height]
    center_roi = roi[:, 0:2]
    center_gt = gt[:, 0:2]
    cpe = numpy.sqrt(numpy.sum((center_gt - center_roi) ** 2, axis=1))
    return cpe

'''
calculate the overlap ratio between two rois
im, shape (3, height, width)
roi, gt, shape (batch_size, 4), normalized version in [0, 1]
'''
def cal_rect_int(roi, gt):

    #roi = roi * [im_width, im_height, im_width, im_height]
    #gt = gt * [im_width, im_height, im_width, im_height]
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
    rect = numpy.loadtxt(os.path.join('D:/HKUST/2-2/learning-to-track/datasets/OTB100/Basketball', 'groundtruth_rect.txt'), delimiter=',')
    im = cv2.imread('D:/HKUST/2-2/learning-to-track/datasets/OTB100/Basketball/img/0001.jpg')
    im_height = im.shape[0]
    im_width = im.shape[1]
    rect[:, 0:2] += rect[:, 2:4]/2 - 1

    rect[:, ::2] = rect[:, ::2] / im.shape[1]
    rect[:, 1::2] = rect[:, 1::2] / im.shape[0]
    A = rect[0:10, :]
    B = rect[10:20, :]
    overlap = cal_rect_int(A, B)
    print overlap
    cpe = calc_CPE(im_height, im_width, A, B)
    print cpe
    overlap = cal_rect_int(A, A)
    print overlap
    cpe = calc_CPE(im_height, im_width, A, A)
    print cpe

