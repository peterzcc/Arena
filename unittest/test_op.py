import numpy as np
from mxnet.test_utils import *
from arena.ops import *
from arena.utils import *

def test_entropy_multinomial():
    dat = np.random.normal(size=(20, 10))
    data = mx.sym.Variable('data')
    data = mx.sym.SoftmaxActivation(data)
    sym = entropy_multinomial(data)
    check_numeric_gradient(sym=sym, location=[dat], numeric_eps=1E-3)


if __name__ == '__main__':
    test_entropy_multinomial()