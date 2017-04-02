import numpy as np
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

if __name__ == '__main__':
    test_entropy_multinomial()
    test_caffe_compatible_pooling()
