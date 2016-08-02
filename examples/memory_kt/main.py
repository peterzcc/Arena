import logging
import argparse
import numpy as np
from load_data import DATA
from arena import Base
from model import MODEL
from arena.utils import *
from run import train
import mxnet.ndarray as nd

from numpy import linalg as LA

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test MANN.')
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--batch_size', type=int, default=10, help='the batch size')
    parser.add_argument('--embed_dim', type=int, default=6, help='embedding dimensions')
    parser.add_argument('--control_state_dim', type=int, default=7, help='hidden states of the controller')
    parser.add_argument('--memory_size', type=int, default=128, help='memory size')
    parser.add_argument('--memory_state_dim', type=int, default=8, help='internal state dimension')
    parser.add_argument('--k_smallest', type=int, default=2, help='parmeter of k smallest flags')

    parser.add_argument('--max_iter', type=int, default=10, help='number of iterations')
    parser.add_argument('--num_reads', type=int, default=1, help='number of read tensors')
    parser.add_argument('--num_writes', type=int, default=1, help='number of write tensors')

    parser.add_argument('--n_question', type=int, default=111, help='the number of unique questions in the dataset')
    parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')

    parser.add_argument('--init_std', type=float, default=0.05, help='weight initialization std')
    parser.add_argument('--init_lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum rate')
    parser.add_argument('--maxgradnorm', type=float, default=50, help='maximum gradient norm')

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
    data_shapes = {'data': (params.seqlen, params.batch_size),
                   'target': (params.seqlen, params.batch_size),
                   'init_memory': (params.batch_size, params.memory_size, params.memory_state_dim),
                   'MANN->write_head:init_W_r_focus': (params.batch_size, params.num_writes, params.memory_size),
                   'MANN->write_head:init_W_u_focus': (params.batch_size, params.num_writes, params.memory_size),
                   'controller->layer0:init_h': (params.batch_size, params.control_state_dim),
                   'controller->layer0:init_c': (params.batch_size, params.control_state_dim)}
    initializer = LRUAInitializer(sigma=params.init_std)
    net = Base(sym_gen=g_model.sym_gen(),
               data_shapes=data_shapes,
               initializer=initializer,
               ctx=ctx,
               name="LRUA_KT"#,
               #default_bucket_kwargs={'seqlen': params.seqlen}
               )
    print "net.params.items()================================================="
    for k, v in net.params.items():
        print k, "\t\t", LA.norm(v.asnumpy())
    #print "==========================================================================="
    #for k, v in net.params.items():
    #    print k, "\n", v.asnumpy()
    #    print "                                                                         ---->",\
    #        k, nd.norm(v).asnumpy()
    #    print "                                                                         ---->", \
    #        k, np.sqrt((v.asnumpy() ** 2).sum())
    #    print "                                                                         ---->", \
    #        k, LA.norm(v.asnumpy())
    #print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    net.print_stat()

    # run -train
    if not params.test:
        for idx in xrange(params.max_iter):
            train_loss = train(net, params, train_data, label='Train')
            #valid_loss = test(net, params, valid_data, label='Validation')

            # logging for each epoch
            m = len(g_log_cost) + 1
            #g_log_cost[m] = [m, train_loss, valid_loss]
            g_log_cost[m] = [m, train_loss]
            output_state = {'epoch': idx + 1,
                            "train_loss": train_loss,
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

