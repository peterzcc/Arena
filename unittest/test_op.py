import numpy as np
import scipy
from scipy.ndimage import interpolation
from mxnet.test_utils import *
from arena.ops import *
from arena.utils import *


def test_entropy_multinomial():
    dat = np.random.normal(size=(20, 10))
    data = mx.sym.Variable('data')
    data = mx.sym.SoftmaxActivation(data)
    sym = entropy_multinomial(data)
    check_numeric_gradient(sym=sym, location=[dat], numeric_eps=1E-3, check_eps=0.02)


def test_caffe_compatible_pooling():
    data = mx.sym.Variable('data')
    pooled_value = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2),
                                  pool_type="max", pooling_convention="full")
    for i in range(5):
        check_numeric_gradient(sym=pooled_value, location=[np.random.normal(size=(2, 3, 11, 11))],
                               grad_nodes={'data': 'add'}, numeric_eps=1E-3, check_eps=0.02)
        check_numeric_gradient(sym=pooled_value, location=[np.random.normal(size=(2, 3, 11, 11))],
                               grad_nodes={'data': 'write'}, numeric_eps=1E-3, check_eps=0.02)


def test_roi_wrapping():
    def calc_gt(data, rois, out_shape, explicit_batch, antialiasing, sampler_typ="bilinear"):
        import cv2
        ret = np.empty([rois.shape[0], data.shape[1], out_shape[0], out_shape[1]], dtype=np.float32)
        for i in range(rois.shape[0]):
            for j in range(data.shape[1]):
                if explicit_batch:
                    ret[i, j, :, :] = cv2.resize(data[int(rois[i, 0]), j].T, dsize=(out_shape[0], out_shape[1])).T
                else:
                    ret[i, j, :, :] = cv2.resize(data[i, j].T, dsize=(out_shape[0], out_shape[1])).T
        return ret
    def ele_test_forward_roiwrapping(data_shape, out_shape, explicit_batch, anti_aliasing,
                                     spatial_scale=1.0, sampler_typ="bilinear", ctx=mx.gpu()):
        data_npy = np.random.rand(*data_shape)
        if explicit_batch:
            rois_num = 3 * data_shape[0]
            rois_npy = np.zeros((rois_num, 5), dtype=np.float32)
            rois_npy[:, 0] = np.random.randint(0, data_shape[0], size=rois_num)
            rois_npy[:, 1:3] = 0
            rois_npy[:, 3] = (float(data_shape[3]) - 1.0) / spatial_scale
            rois_npy[:, 4] = (float(data_shape[2]) - 1.0) / spatial_scale
        else:
            rois_num = data_shape[0]
            rois_npy = np.zeros((rois_num, 4), dtype=np.float32)
            rois_npy[:, 0:2] = 0
            rois_npy[:, 2] = (float(data_shape[3]) - 1.0) / spatial_scale
            rois_npy[:, 3] = (float(data_shape[2]) - 1.0) / spatial_scale
        gt = calc_gt(data=data_npy, rois=rois_npy, out_shape=out_shape,
                     explicit_batch=explicit_batch, antialiasing=False, sampler_typ='bilinear')
        data = mx.sym.Variable('data')
        rois = mx.sym.Variable('rois')
        sym = mx.sym.ROIWrapping(data=data, rois=rois, pooled_size=out_shape, interp_typ=sampler_typ,
                                 spatial_scale=spatial_scale, explicit_batch=explicit_batch)
        exe = sym.simple_bind(ctx=ctx, data=data_npy.shape, rois=rois_npy.shape)
        outputs = exe.forward(is_train=False, data=data_npy, rois=rois_npy)
        #print data_npy
        #print outputs[0].asnumpy()
        #print gt
        assert reldiff(outputs[0].asnumpy(), gt) < 1E-6, "diff=%s, out=%s, gt=%s"\
                                                         %(str(outputs[0].asnumpy() - gt),
                                                           str(outputs[0].asnumpy()),
                                                           str(gt))
    def ele_test_backward_roiwrapping(data_shape, out_shape, explicit_batch, anti_aliasing,
                                      spatial_scale=1.0, sampler_typ="bilinear", ctx=mx.gpu()):
        data = mx.sym.Variable('data')
        rois = mx.sym.Variable('rois')
        sym = mx.sym.ROIWrapping(data=data, rois=rois, pooled_size=out_shape, anti_aliasing=anti_aliasing,
                                 spatial_scale=spatial_scale, explicit_batch=explicit_batch, interp_typ=sampler_typ)
        #data_npy = np.random.rand(*data_shape)
        data_npy = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 2.0], [1.0, 2.0, 1.0]]).reshape((1, 1, 3, 3))
        if explicit_batch:
            rois_num = 3 * data_shape[0]
            rois_npy = np.zeros((rois_num, 5), dtype=np.float32)
            rois_npy[:, 0] = np.random.randint(0, data_shape[0], size=rois_num)
            rois_npy[:, 1] = np.random.rand(rois_num) * (data_shape[3] - 1.0) / spatial_scale
            rois_npy[:, 2] = np.random.rand(rois_num) * (data_shape[2] - 1.0) / spatial_scale
            rois_npy[:, 3] = np.random.rand(rois_num) * ((data_shape[3] - 1.0) / spatial_scale
                                                         - rois_npy[:, 1]) + rois_npy[:, 1]
            rois_npy[:, 4] = np.random.rand(rois_num) * ((data_shape[2] - 1.0) / spatial_scale
                                                         - rois_npy[:, 2]) + rois_npy[:, 2]
        else:
            rois_num = data_shape[0]
            rois_npy = np.zeros((rois_num, 4), dtype=np.float32)
            rois_npy[:, 0] = np.random.rand(rois_num) * (data_shape[3] - 1.0) / spatial_scale
            rois_npy[:, 1] = np.random.rand(rois_num) * (data_shape[2] - 1.0) / spatial_scale
            rois_npy[:, 2] = np.random.rand(rois_num) * ((data_shape[3] - 1.0) / spatial_scale
                                                         - rois_npy[:, 0]) + rois_npy[:, 0]
            rois_npy[:, 3] = np.random.rand(rois_num) * ((data_shape[2] - 1.0) / spatial_scale
                                                         - rois_npy[:, 1]) + rois_npy[:, 1]
        rois_npy = np.array([[0.1, 0.1, 1.2, 1.2]])
        print data_npy
        print rois_npy
        exe = sym.simple_bind(ctx=ctx, data=data_npy.shape, rois=rois_npy.shape)
        outputs = exe.forward(is_train=False, data=data_npy, rois=rois_npy)
        print outputs[0].asnumpy()
        check_numeric_gradient(sym, location={'data': data_npy, 'rois': rois_npy},
                               grad_nodes={'data': 'write', 'rois':'null'}, numeric_eps=1E-2,
                               check_eps=0.05,
                               ctx=ctx)
        check_numeric_gradient(sym, location={'data': data_npy, 'rois': rois_npy},
                               grad_nodes={'data': 'add', 'rois': 'null'}, numeric_eps=1E-2,
                               check_eps=0.05,
                               ctx=ctx)
        check_numeric_gradient(sym, location={'data': data_npy, 'rois': rois_npy},
                               grad_nodes={'rois': 'write', 'data': 'null'}, numeric_eps=1E-4,
                               check_eps=0.05,
                               ctx=ctx)
        check_numeric_gradient(sym, location={'data': data_npy, 'rois': rois_npy},
                               grad_nodes={'rois': 'add', 'data': 'null'}, numeric_eps=1E-4,
                               check_eps=0.05,
                               ctx=ctx)
    for ctx in [mx.cpu(), mx.gpu()]:
        for spatial_scale in [0.1, 0.3, 1.0]:
            ele_test_forward_roiwrapping(data_shape=(1, 1, 2, 2), out_shape=(3, 3),
                                         explicit_batch=True,
                                         spatial_scale=spatial_scale, anti_aliasing=False, ctx=ctx,
                                         sampler_typ="bicubic")
            ele_test_forward_roiwrapping(data_shape=(1, 1, 2, 2), out_shape=(3, 3), explicit_batch=True,
                                         spatial_scale=spatial_scale, anti_aliasing=False, ctx=ctx)
            ele_test_forward_roiwrapping(data_shape=(1, 1, 2, 4), out_shape=(3, 2), explicit_batch=False,
                                         spatial_scale=spatial_scale, anti_aliasing=False, ctx=ctx)
            ele_test_forward_roiwrapping(data_shape=(1, 1, 2, 4), out_shape=(3, 5), explicit_batch=True,
                                         spatial_scale=spatial_scale, anti_aliasing=False, ctx=ctx)
            ele_test_forward_roiwrapping(data_shape=(1, 1, 23, 2), out_shape=(2, 23), explicit_batch=False,
                                         spatial_scale=spatial_scale, anti_aliasing=False, ctx=ctx)
            ele_test_forward_roiwrapping(data_shape=(1, 1, 43, 22), out_shape=(23, 43), explicit_batch=False,
                                         spatial_scale=spatial_scale, anti_aliasing=False, ctx=ctx)
    for ctx in [mx.gpu()]:
        for spatial_scale in [1.0]:
            print spatial_scale
#            ele_test_backward_roiwrapping(data_shape=(5, 3, 7, 7), out_shape=(3, 3), explicit_batch=True,
#                                          spatial_scale=spatial_scale, anti_aliasing=False, ctx=ctx)
#            ele_test_backward_roiwrapping(data_shape=(5, 3, 3, 3), out_shape=(5, 4),
#                                          explicit_batch=False,
#                                          spatial_scale=spatial_scale, anti_aliasing=False, ctx=ctx)
            ele_test_backward_roiwrapping(data_shape=(1, 1, 3, 3), out_shape=(1, 1),
                                          explicit_batch=False,
                                          spatial_scale=spatial_scale, anti_aliasing=True, ctx=ctx)


if __name__ == '__main__':
    test_entropy_multinomial()
    test_caffe_compatible_pooling()
    test_roi_wrapping()