import logging
import argparse
import sys
import math
import mxnet as mx
import numpy as np

from load_data import DATA
from arena import Base
from model import MODEL
from arena.utils import *

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

class LRUAInitializer(mx.initializer.Normal):
    def _init_weight(self, name, arr):
        super(LRUAInitializer, self)._init_weight(name, arr)


def train(net, params, data, label):
    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    np.random.shuffle(data)
    N = int(math.floor(len(data) / params.batch_size))
    cost = 0

    #one_seq = np.ndarray([params.batch_size, params.seqlen], dtype=np.float32)
    #input_x = np.ndarray([params.batch_size, params.seqlen-1], dtype=np.float32)
    #target = np.ndarray([params.batch_size, params.seqlen-1], dtype=np.float32)

    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)
    init_memory_npy = np.tanh(np.random.normal(size=(params.batch_size, params.memory_size, params.memory_state_dim)))
    init_h_npy = np.zeros((params.batch_size, params.control_state_dim),
                             dtype=np.float32) + 0.0001  # numpy.tanh(numpy.random.normal(size=(batch_size, control_state_dim)))
    init_c_npy = np.zeros((params.batch_size, params.control_state_dim),
                             dtype=np.float32) + 0.0001  # numpy.tanh(numpy.random.normal(size=(batch_size, control_state_dim)))
    init_read_focus_npy = npy_softmax(
        np.broadcast_to(np.arange(params.memory_size, 0, -1), (params.batch_size, params.memory_size)),
        axis=1)
    init_write_focus_npy = npy_softmax(
        np.broadcast_to(np.arange(params.memory_size, 0, -1), (params.batch_size, params.memory_size)),
        axis=1)
    for idx in xrange(N):
        if params.show: bar.next()
        one_seq = data[idx*params.batch_size : (idx+1)*params.batch_size ]
        input_x = one_seq[:,:-1]
        target = one_seq[:,1:]
        # data_shapes = {'data': (params.batch_size, params.seqlen),
        #           'target': (params.batch_size, params.seqlen),
        #           'init_memory': (params.batch_size, params.memory_size, params.memory_state_dim),
        #           'read_init_focus': (params.batch_size, params.memory_size),
        #           'write_init_focus': (params.batch_size, params.memory_size),
        #           'controller_init_h': (params.batch_size, params.control_state_dim),
        #           'controller_init_c': (params.batch_size, params.control_state_dim)},
        outputs = net.forward(is_train=True,
                          data=input_x, target=target,
                          init_memory=init_memory_npy,
                          read_init_focus=init_read_focus_npy, # TODO need to change
                          write_init_focus=init_write_focus_npy, # TODO need to change
                          controller_init_h=init_h_npy,
                          controller_init_c=init_c_npy)
        net.backward()
        norm_clipping(net.params_grad, params.maxgradnorm)
        optimizer = mx.optimizer.create(name='SGD', learning_rate=params.lr, momentum=params.momentum,
                                        # rescale_grad=1.0 / params.batch_size)
                                        rescale_grad=1.0)
        updater = mx.optimizer.get_updater(optimizer)
        net.update(updater=updater)

        ### print parameter information
        for k, v in net.params_grad.items():
            print k, nd.norm(v).asnumpy()

        ### get results and compute the loss
        pred = outputs[0].\
            reshape((params.seqlen, params.batch_size, params.data_dim)).asnumpy() # TODO need to change
        avg_loss = npy_binary_entropy(pred, target) / params.seqlen # TODO need to change
        cost += avg_loss
        print avg_loss
    if params.show: bar.finish()

    one_epoch_loss = cost / N / params.batch_size
    print label, "loss:", one_epoch_loss
    return one_epoch_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test MANN.')
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size')
    parser.add_argument('--embed_dim', type=int, default=100, help='embedding dimensions')
    parser.add_argument('--control_state_dim', type=int, default=100, help='hidden states of the controller')
    parser.add_argument('--memory_size', type=int, default=128, help='memory size')
    parser.add_argument('--memory_state_dim', type=int, default=20, help='internal state dimension')

    parser.add_argument('--max_iter', type=int, default=10000, help='number of iterations')
    parser.add_argument('--num_reads', type=int, default=1, help='number of read tensors')
    parser.add_argument('--num_writes', type=int, default=1, help='number of write tensors')

    parser.add_argument('--n_question', type=int, default=111, help='the number of unique questions in the dataset')
    parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')

    parser.add_argument('--init_std', type=float, default=0.05, help='weight initialization std')
    parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum rate')
    parser.add_argument('--maxgradnorm', type=float, default=10, help='maximum gradient norm')

    parser.add_argument('--test', type=bool, default=False, help='enable testing')
    parser.add_argument('--show', type=bool, default=True, help='print progress')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory [data]')
    parser.add_argument('--data_name', type=str, default='builder', help='data set name [ptb]')
    #parser.add_argument('--load', type=str, default='MemNN', help='model file to load')
    #parser.add_argument('--save', type=str, default='MemNN', help='path to save model')
    params = parser.parse_args()



    # Reading data
    dat = DATA(n_question = params.n_question, seqlen=params.seqlen, separate_char=',')
    train_data_path = params.data_dir + "/" + params.data_name + "_train.csv"
    test_data_path = params.data_dir + "/" + params.data_name + "_test.csv"
    train_data = dat.load_data(train_data_path)
    test_data = dat.load_data(test_data_path)
    print "train_data.shape",train_data.shape ###(3633, 200) = (#sample, seqlen)
    print "test_data.shape",test_data.shape   ###(1566, 200)

    g_log_cost = {}

    params.lr = params.init_lr

    # choose ctx
    if params.gpus == None:
        ctx = mx.cpu()
        print "Training with cpu ..."
    else:
        ctx = mx.gpu(int(params.gpus))
        print "Training with gpu(" + params.gpus + ") ..."

    # model
    ### def __init__(self, n_question, seqlen,
    ###              embed_dim, control_state_dim, memory_size, memory_state_dim, k_smallest,
    ###              num_reads, num_writes,
    ###              name="KT"):
    g_model = MODEL(n_question = params.n_question,
                    seqlen = params.seqlen,
                    embed_dim = params.embed_dim,
                    control_state_dim = params.control_state_dim,
                    memory_size = params.memory_size,
                    memory_state_dim = params.memory_state_dim,
                    k_smallest = params.k_smallest,
                    num_reads = params.num_reads,
                    num_writes = params.num_writes)
    # train model
    data_shapes = {'data': (params.batch_size, params.seqlen),
                   'target': (params.batch_size, params.seqlen),
                   'init_memory': (params.batch_size, params.memory_size, params.memory_state_dim),
                   #'read_init_focus': (params.batch_size, params.memory_size),
                   #'write_init_focus': (params.batch_size, params.memory_size),
                   'controller_init_h': (params.batch_size, params.control_state_dim),
                   'controller_init_c': (params.batch_size, params.control_state_dim)},
    initializer = LRUAInitializer(sigma=params.init_std)
    net = Base(sym_gen=g_model.sym_gen(),
               data_shapes=data_shapes,
               initializer=initializer,
               ctx=ctx,
               name="LRUA_KT")
    net.print_stat()
    # run -train
    if not params.test:
        for idx in xrange(params.nepoch):
            train_loss = train(net, params, train_data, label='Train')
            #valid_loss = test(net, params, valid_data, label='Validation')

            # logging for each epoch
            m = len(g_log_cost) + 1
            #g_log_cost[m] = [m, train_loss, valid_loss]
            g_log_cost[m] = [m, train_loss]
            output_state = {'epoch': idx + 1,
                            "train_perplexity": train_loss,
                            #"valid_perplexity": np.exp(valid_loss),
                            "learning_rate": params.lr}
            print output_state

            # Learning rate annealing
            if m > 1 and g_log_cost[m][2] > g_log_cost[m - 1][2] * 0.9999:
                params.lr = params.lr / 1.5
            if params.lr < 1e-5: break

        net.save_params(dir_path=os.path.join('model', params.save))
        print g_log_cost
    # run -test

