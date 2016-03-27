import mxnet as mx
import mxnet.ndarray as nd
from mxnet import kvstore
import numpy
from arena import Base
from arena.games import AtariGame
from arena.utils import *
import logging
import argparse
import os
import re
import sys
import time

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
mx.random.seed(100)
npy_rng = get_numpy_rng()


# TODO NDArrayOP will cause some troubles see `https://github.com/dmlc/mxnet/issues/1720'
class DQNOutputOp(mx.operator.NDArrayOp):
    def __init__(self):
        super(DQNOutputOp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'action', 'reward']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_shape = (in_shape[0][0],)
        reward_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, reward_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = x

    def backward(self, out_grad, in_data, out_data, in_grad):
        x = out_data[0]
        action = in_data[1]
        reward = in_data[2]
        dx = in_grad[0]
        dx[:] = 0
        dx[:] = nd.fill_element_0index(dx,
                                       nd.clip(nd.choose_element_0index(x, action) - reward, -1, 1),
                                       action)

def update_to_kvstore(kvStore,params,params_grad):
    for paramIndex in range(len(params)):
        k=params.keys()[paramIndex]
        kvStore.push(paramIndex,params_grad[k],priority=-paramIndex)
        timeAfterPush = time.time()
        kvStore.pull(paramIndex,params[k],priority=-paramIndex)
# TODO Regression Output has none differential for label, we may need to fix that
class DQNOutputNpyOp(mx.operator.NumpyOp):
    def __init__(self):
        super(DQNOutputNpyOp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'action', 'reward']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_shape = (in_shape[0][0],)
        reward_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, reward_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = x

    def backward(self, out_grad, in_data, out_data, in_grad):
        x = out_data[0]
        action = in_data[1].astype(numpy.int)
        reward = in_data[2]
        dx = in_grad[0]
        dx[:] = 0
        dx[numpy.arange(action.shape[0]), action] \
            = numpy.clip(x[numpy.arange(action.shape[0]), action] - reward, -1, 1)


def dqn_sym_nips(action_num, output_op):
    net = mx.symbol.Variable('data')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=16)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc3', num_hidden=256)
    net = mx.symbol.Activation(data=net, name='relu3', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=action_num)
    net = output_op(data=net, name='dqn')
    return net


def dqn_sym_nature(action_num, output_op):
    net = mx.symbol.Variable('data')
    net = mx.symbol.Convolution(data=net, name='conv1', kernel=(8, 8), stride=(4, 4), num_filter=32)
    net = mx.symbol.Activation(data=net, name='relu1', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv2', kernel=(4, 4), stride=(2, 2), num_filter=64)
    net = mx.symbol.Activation(data=net, name='relu2', act_type="relu")
    net = mx.symbol.Convolution(data=net, name='conv3', kernel=(3, 3), stride=(1, 1), num_filter=64)
    net = mx.symbol.Activation(data=net, name='relu3', act_type="relu")
    net = mx.symbol.Flatten(data=net)
    net = mx.symbol.FullyConnected(data=net, name='fc4', num_hidden=512)
    net = mx.symbol.Activation(data=net, name='relu4', act_type="relu")
    net = mx.symbol.FullyConnected(data=net, name='fc5', num_hidden=action_num)
    net = output_op(data=net, name='dqn')
    return net


class DQNInitializer(mx.initializer.Xavier):
    def _init_bias(self, _, arr):
        arr[:] = .1


def main():
    parser = argparse.ArgumentParser(description='Script to test the trained network on a game.')
    parser.add_argument('-r', '--rom', required=False, type=str,
                        default=os.path.join('arena', 'games', 'roms', 'breakout.bin'),
                        help='Path of the ROM File.')
    parser.add_argument('-v', '--visualization', required=False, type=int, default=0,
                        help='Visualize the runs.')
    parser.add_argument('--lr', required=False, type=float, default=0.01,
                        help='Learning rate of the AdaGrad optimizer')
    parser.add_argument('--eps', required=False, type=float, default=0.01,
                        help='Eps of the AdaGrad optimizer')
    parser.add_argument('--clip-gradient', required=False, type=float, default=None,
                        help='Clip threshold of the AdaGrad optimizer')
    parser.add_argument('--double-q', required=False, type=bool, default=False,
                        help='Use Double DQN')
    parser.add_argument('--wd', required=False, type=float, default=0.0,
                        help='Weight of the L2 Regularizer')
    parser.add_argument('-c', '--ctx', required=False, type=str, default='gpu',
                        help='Running Context. E.g `-c gpu` or `-c gpu1` or `-c cpu`')
    parser.add_argument('-d', '--dir-path', required=False, type=str, default='',
                        help='Saving directory of model files.')
    parser.add_argument('--start-eps', required=False, type=float, default=1.0,
                        help='Eps of the epsilon-greedy policy at the beginning')
    parser.add_argument('--replay-start-size', required=False, type=int, default=50000,
                        help='The step that the training starts')
    parser.add_argument('--kvstore-update-period', required=False, type=int, default=1,
                        help='The period that the worker updates the parameters from the sever')
    parser.add_argument('--kv-type', required=False, type=str, default=None,
                        help='type of kvstore, default will not use kvstore, could also be dist_async')
    args, unknown = parser.parse_known_args()
    if args.dir_path == '':
        rom_name = os.path.splitext(os.path.basename(args.rom))[0]
        args.dir_path = 'dqn-%s' % rom_name
    ctx = re.findall('([a-z]+)(\d*)', args.ctx)
    ctx = [(device, int(num)) if len(num) >0 else (device, 0) for device, num in ctx]

    q_ctx = mx.Context(*ctx[0])


    testShape = (1,1)
    testParam = nd.ones(testShape,ctx=q_ctx)
    testGrad = nd.zeros(testShape,ctx=q_ctx)
    if args.kv_type != None:
        kvType = args.kv_type
        kvStore = kvstore.create(kvType)
        kvStore.init(0,testParam)
        optimizer = mx.optimizer.create(name='sgd', learning_rate=1.0, wd=args.wd)
        kvStore.set_optimizer(optimizer)
        kvstore_update_period = args.kvstore_update_period

    while (1):
        time_before_wait = time.time()
        kvStore.push(0,testGrad,priority=0)
        kvStore.pull(0,testParam,priority=0)
        testParam.wait_to_read()
        logging.info("wait time %f" %(time.time()-time_before_wait))


if __name__ == '__main__':
    main()
