import numpy as np
import scipy
from scipy.ndimage import interpolation
from mxnet.test_utils import *
from arena.ops import *
from arena.utils import *

np.random.seed(123)
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
        resize_typ = cv2.INTER_CUBIC if sampler_typ == "bicubic" else cv2.INTER_LINEAR
        ret = np.empty([rois.shape[0], data.shape[1], out_shape[0], out_shape[1]], dtype=np.float32)
        for i in range(rois.shape[0]):
            for j in range(data.shape[1]):
                if explicit_batch:
                    ret[i, j, :, :] = cv2.resize(data[int(rois[i, 0]), j].T, dsize=(out_shape[0], out_shape[1]),
                                                 interpolation=resize_typ).T
                else:
                    ret[i, j, :, :] = cv2.resize(data[i, j].T, dsize=(out_shape[0], out_shape[1]),
                                                 interpolation=resize_typ).T
        return ret
    def ele_test_forward_roiwrapping(data_shape, out_shape, explicit_batch, anti_aliasing,
                                     spatial_scale=1.0, sampler_typ="bilinear", ctx=mx.gpu()):
        data_npy = np.random.rand(*data_shape)
        #data_npy = np.array([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]]], dtype=np.float32)
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
                     explicit_batch=explicit_batch, antialiasing=False, sampler_typ=sampler_typ)
        data = mx.sym.Variable('data')
        rois = mx.sym.Variable('rois')
        sym = mx.sym.ROIWrapping(data=data, rois=rois, pooled_size=out_shape, interp_type=sampler_typ,
                                 spatial_scale=spatial_scale, explicit_batch=explicit_batch)
        exe = sym.simple_bind(ctx=ctx, data=data_npy.shape, rois=rois_npy.shape)
        outputs = exe.forward(is_train=False, data=data_npy, rois=rois_npy)
        # print 'data:', data_npy
        # print 'outputs:', outputs[0].asnumpy()
        # print 'gt:', gt
        assert reldiff(outputs[0].asnumpy(), gt) < 1E-6, "diff=%s, out=%s, gt=%s"\
                                                         %(str(outputs[0].asnumpy() - gt),
                                                           str(outputs[0].asnumpy()),
                                                           str(gt))
    def ele_test_backward_roiwrapping(data_shape, out_shape, explicit_batch, anti_aliasing,
                                      spatial_scale=1.0, sampler_typ="bilinear", ctx=mx.gpu()):
        data = mx.sym.Variable('data')
        rois = mx.sym.Variable('rois')
        sym = mx.sym.ROIWrapping(data=data, rois=rois, pooled_size=out_shape, anti_aliasing=anti_aliasing,
                                 spatial_scale=spatial_scale, explicit_batch=explicit_batch,
                                 interp_type=sampler_typ)
        data_npy = np.random.rand(*data_shape)
        #data_npy = np.array([[2, 1, 2], [1, 2, 1], [2, 1, 2]]).reshape((1, 1, 3, 3))
        rois_npy = None
        if explicit_batch:
            rois_num = 5 * data_shape[0]
            rois_npy = np.zeros((rois_num, 5), dtype=np.float32)
            rois_npy[:, 0] = np.random.randint(0, data_shape[0], size=rois_num).astype(np.float32)
            rois_npy[:, 1] = (np.random.rand(rois_num)-0.1) * (data_shape[3] - 1.0) / spatial_scale
            rois_npy[:, 2] = (np.random.rand(rois_num)-0.1) * (data_shape[2] - 1.0) / spatial_scale
            rois_npy[:, 3] = (np.random.rand(rois_num)) * ((data_shape[3] - 1.0) / spatial_scale
                                                         - rois_npy[:, 1]) + rois_npy[:, 1]
            rois_npy[:, 4] = (np.random.rand(rois_num)) * ((data_shape[2] - 1.0) / spatial_scale
                                                         - rois_npy[:, 2]) + rois_npy[:, 2]
            # rois_npy[:, 1] = 0.1
            # rois_npy[:, 2] = 0.1
            # rois_npy[:, 3] = 1.2
            # rois_npy[:, 4] = 1.2
            # assert rois_npy.shape[1] == 5
            # print rois_npy
        else:
            rois_num = data_shape[0]
            rois_npy = np.zeros((rois_num, 4), dtype=np.float32)
            rois_npy[:, 0] = np.random.rand(rois_num) * (data_shape[3] - 1.0) / spatial_scale
            rois_npy[:, 1] = np.random.rand(rois_num) * (data_shape[2] - 1.0) / spatial_scale
            rois_npy[:, 2] = np.random.rand(rois_num) * ((data_shape[3] - 1.0) / spatial_scale
                                                         - rois_npy[:, 0]) + rois_npy[:, 0]
            rois_npy[:, 3] = np.random.rand(rois_num) * ((data_shape[2] - 1.0) / spatial_scale
                                                         - rois_npy[:, 1]) + rois_npy[:, 1]
        #     rois_npy[:, 0] = 0.1
        #     rois_npy[:, 1] = 0.1
        #     rois_npy[:, 2] = 1.2
        #     rois_npy[:, 3] = 1.2
        # print data_npy
        # print rois_npy
        check_numeric_gradient(sym, location={'data': data_npy, 'rois': rois_npy},
                               grad_nodes={'data': 'write', 'rois':'null'}, numeric_eps=1E-2,
                               check_eps=0.01,
                               ctx=ctx)
        check_numeric_gradient(sym, location={'data': data_npy, 'rois': rois_npy},
                               grad_nodes={'data': 'add', 'rois': 'null'}, numeric_eps=1E-2,
                               check_eps=0.01,
                               ctx=ctx)
        check_numeric_gradient(sym, location={'data': data_npy, 'rois': rois_npy},
                               grad_nodes={'rois': 'write', 'data': 'null'}, numeric_eps=1E-2,
                               check_eps=0.1,
                               ctx=ctx)
        check_numeric_gradient(sym, location={'data': data_npy, 'rois': rois_npy},
                               grad_nodes={'rois': 'add', 'data': 'null'}, numeric_eps=1E-2,
                               check_eps=0.1,
                               ctx=ctx)
    def ele_check_speed(data_shape, out_shape, roi_num, spatial_scale, ctx=mx.gpu()):
        data = mx.sym.Variable('data')
        rois = mx.sym.Variable('rois')
        data_npy = np.random.rand(*data_shape)
        #data_npy = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 2.0], [1.0, 2.0, 1.0]]).reshape((1, 1, 3, 3))
        rois_num = roi_num
        rois_npy = np.zeros((rois_num, 5), dtype=np.float32)
        rois_npy[:, 0] = np.random.randint(0, data_shape[0], size=rois_num)
        rois_npy[:, 1] = np.random.rand(rois_num) * (data_shape[3] - 1.0) / spatial_scale
        rois_npy[:, 2] = np.random.rand(rois_num) * (data_shape[2] - 1.0) / spatial_scale
        rois_npy[:, 3] = np.random.rand(rois_num) * ((data_shape[3] - 1.0) / spatial_scale
                                                     - rois_npy[:, 1]) + rois_npy[:, 1]
        rois_npy[:, 4] = np.random.rand(rois_num) * ((data_shape[2] - 1.0) / spatial_scale
                                                     - rois_npy[:, 2]) + rois_npy[:, 2]
        for sampler_typ in ["max", "bilinear", "bicubic"]:
            for anti_aliasing in [False, True]:
                if sampler_typ != "max":
                    sym = mx.sym.ROIWrapping(data=data, rois=rois, pooled_size=out_shape, anti_aliasing=anti_aliasing,
                                             spatial_scale=spatial_scale, explicit_batch=True,
                                             interp_type=sampler_typ)
                else:
                    sym = mx.sym.ROIPooling(data=data, rois=rois, pooled_size=out_shape, spatial_scale=spatial_scale)
                forward_time = check_speed(sym, location={'data': data_npy, 'rois': rois_npy}, N=10, typ="forward", ctx=ctx)
                forward_backward_time = check_speed(sym, location={'data': data_npy, 'rois': rois_npy}, N=10, typ="whole", ctx=ctx)
                if sampler_typ != "max":
                    print("ctx:%s, data_shape:%s, out_shape:%s, anti_aliasing:%s, sampler:%s, f:%f, fb:%f"
                          %(str(ctx), str(data_shape), str(out_shape), str(anti_aliasing), sampler_typ, forward_time, forward_backward_time))
                else:
                    print("ctx:%s, data_shape:%s, out_shape:%s, sampler:%s, f:%f, fb:%f"
                          % (str(ctx), str(data_shape), str(out_shape), sampler_typ,
                             forward_time, forward_backward_time))

    for ctx in [mx.cpu(), mx.gpu()]:
        for spatial_scale in [0.01, 0.5, 1.0]:
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
        for spatial_scale in [0.5, 1.0]:
            for sampler_typ in ["bilinear","bicubic"]:
                for anti_aliasing in [True, False]:
                    for explicit_batch in [False, True]:
                        print("ctx:", ctx, "spatial_scale:", spatial_scale, "sampler_typ:", sampler_typ, "explicit_batch:", explicit_batch, "anti_aliasing:", anti_aliasing)
                        ele_test_backward_roiwrapping(data_shape=(2, 3, 7, 7), out_shape=(3, 3),
                                                      explicit_batch=False,
                                                      spatial_scale=spatial_scale, anti_aliasing=anti_aliasing,
                                                      sampler_typ=sampler_typ, ctx=ctx)
                        ele_test_backward_roiwrapping(data_shape=(3, 3, 3, 3), out_shape=(5, 4),
                                                      explicit_batch=True,
                                                      spatial_scale=spatial_scale, anti_aliasing=anti_aliasing,
                                                      sampler_typ=sampler_typ,ctx=ctx)
                        ele_test_backward_roiwrapping(data_shape=(5, 3, 11, 11), out_shape=(3, 3),
                                                      explicit_batch=False,
                                                      spatial_scale=spatial_scale, anti_aliasing=anti_aliasing,
                                                      sampler_typ=sampler_typ, ctx=ctx)
                        ele_test_backward_roiwrapping(data_shape=(1, 2, 23, 23), out_shape=(9, 5),
                                                      explicit_batch=True,
                                                      spatial_scale=spatial_scale, anti_aliasing=anti_aliasing,
                                                      sampler_typ=sampler_typ, ctx=ctx)
    ele_check_speed(data_shape=(2, 512, 32, 32), out_shape=(7, 7), roi_num=128, spatial_scale=0.0625, ctx=mx.gpu())
    ele_check_speed(data_shape=(2, 512, 128, 128), out_shape=(7, 7), roi_num=128, spatial_scale=0.0625, ctx=mx.gpu())

if __name__ == '__main__':
    test_entropy_multinomial()
    test_caffe_compatible_pooling()
    test_roi_wrapping()