import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import matplotlib.pyplot as plt
from lrua import MANN
from arena import Base
from arena.helpers.visualization import *
from arena.ops import LSTM
from arena.utils import *
import logging
import argparse
import sys


root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

# line format
# 15
# 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54,
# 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,
def read_max_n_instance_n_question(path,separate_char=','):
    f_data = open(path,'r')
    max_n_instance = 0
    n_question = 0
    for lineID, line in enumerate(f_data):
        line = line.strip( )
        if lineID % 3 == 0:
            n_instance = int(line)
            if n_instance > max_n_instance:
                max_n_instance = n_instance
        elif lineID % 3 == 1:
            Q = line.split(separate_char)
            for i, q in enumerate(Q):
                if len(q) > 0:
                    questionID = int(q)
                    if questionID > n_question:
                        n_question = questionID
    f_data.close()
    return max_n_instance, n_question

def read_content(path, n_question, set_max_seqlen, separate_char=','):
    f_data = open(path , 'r')
    data = []
    for lineID, line in enumerate(f_data):
        line = line.strip( )
        # lineID starts from 0
        if lineID % 3 == 1:
            Q = line.split(separate_char)
            if len( Q[len(Q)-1] ) == 0:
                Q = Q[:-1]
            #print len(Q)
        elif lineID % 3 == 2:
            A = line.split(separate_char)
            if len( A[len(A)-1] ) == 0:
                A = A[:-1]
            #print len(A),A

            # start split the data
            n_split = 1
            #print 'len(Q):',len(Q)
            if len(Q) > set_max_seqlen:
                n_split = len(Q) / set_max_seqlen
                if len(Q) % set_max_seqlen:
                    n_split = len(Q) / set_max_seqlen + 1
            #print 'n_split:',n_split
            for k in range(n_split):
                instance = []
                if k == n_split - 1:
                    endINdex  = len(A)
                else:
                    endINdex = (k+1) * set_max_seqlen
                for i in range(k * set_max_seqlen, endINdex):
                    if len(Q[i]) > 0 :
                        # int(A[i]) is in {0,1}
                        Xindex = int(Q[i]) + int(A[i]) * n_question
                        instance.append(Xindex)
                    else:
                        print Q[i]
                #print 'instance:-->', len(instance),instance
                data.append(instance)
    f_data.close()
    ### data: [[],[],[],...] <-- set_max_seqlen is used
    print len(data),data

    # data size = (2 * seqlen + 2,  batch_size, data_dim)
    ### convert data into ndarrays for better speed during training
    dataArray = np.zeros((len(data), set_max_seqlen))
    for j in range(len(data)):
        dat = data[j]
        dataArray[j, :len(dat)] = dat
    # dataArray: [ array([[],[],..])]
    return dataArray


class knowledge_tracing(object):
    def __init__(self, data, set_max_seqlen, batch_size, max_iter, num_reads, num_writes,
                 memory_size, memory_state_dim, control_state_dim,
                 init_memory=None, init_read_focus=None, init_write_focus=None, name="KT"):
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.num_reads = num_reads
        self.num_writes = num_writes
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.control_state_dim = control_state_dim

        self.data = data
        self.set_max_seqlen = set_max_seqlen

    def sym_gen(self, set_max_seqlen):
        print set_max_seqlen
        # data_seqlen = 2 * seqlen + 2
        data = mx.sym.Variable('data')
        target = mx.sym.Variable('target')
        #data = mx.sym.SliceChannel(data, num_outputs=data_seqlen, axis=0,
        #                           squeeze_axis=True)  # (batch_size, data_dim) * seqlen
        ### Initialize Memory
        init_memory = mx.sym.Variable('init_memory')
        init_read_focus = [mx.sym.Variable('MANN->read_head%d:init_focus' % i) for i in range(self.num_reads)]
        init_write_focus = [mx.sym.Variable('MANN->write_head%d:init_focus' % i) for i in range(self.num_writes)]
        # Initialize Control Network
        controller = LSTM(num_hidden=self.control_state_dim, name="controller")
        mem = MANN(num_reads=self.num_reads, num_writes=self.num_writes,
                   memory_size=self.memory_size, memory_state_dim=self.memory_state_dim,
                   control_state_dim=self.control_state_dim,
                   init_memory=init_memory,
                   init_read_focus=init_read_focus, init_write_focus=init_write_focus,
                   name="MANN") # TODO init_read_focus / init_write_focus

        controller_h = controller.init_h[0]
        controller_c = controller.init_c[0]

        read_content_l, read_focus_l = mem.read(controller_h)
        write_focus_l = mem.write(controller_h)

        controller_states = []
        all_read_focus_l = []
        all_write_focus_l = []
        #all_read_content_l = []

        for i in range(set_max_seqlen):
            controller_h, controller_c = \
                controller.step(data=mx.sym.Concat(data[i], *read_content_l,
                                                   num_args=1 + len(read_content_l)),
                                prev_h=controller_h,
                                prev_c=controller_c,
                                seq_length=1)
            controller_h = controller_h[0]
            controller_c = controller_c[0]
            read_content_l, read_focus_l = mem.read(controller_h)
            write_focus_l = mem.write(controller_h)
            controller_states.append(controller_h)
            all_read_focus_l.append(read_focus_l[0])
            all_write_focus_l.append(write_focus_l[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test MANN.')
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size')
    parser.add_argument('--memory_size', type=int, default=128, help='memory size')
    parser.add_argument('--memory_state_dim', type=int, default=20, help='internal state dimension')
    parser.add_argument('--control_state_dim', type=int, default=100, help='hidden states of the controller')
    parser.add_argument('--max_iter', type=int, default=10000, help='number of iterations')
    parser.add_argument('--num_reads', type=int, default=1, help='number of read tensors')
    parser.add_argument('--num_writes', type=int, default=1, help='number of write tensors')

    parser.add_argument('--set_max_seqlen', type=int, default=200, help='the allowed maximum length of a sequence')

    #parser.add_argument('--init_hid', type=float, default=0.1, help='initial internal state value')
    #parser.add_argument('--init_std', type=float, default=0.05, help='weight initialization std')
    #parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
    #parser.add_argument('--momentum', type=float, default=0.0, help='momentum rate')
    #parser.add_argument('--maxgradnorm', type=float, default=50, help='maximum gradient norm')

    #parser.add_argument('--test', type=bool, default=False, help='enable testing')
    #parser.add_argument('--show', type=bool, default=True, help='print progress')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory [data]')
    parser.add_argument('--data_name', type=str, default='builder', help='data set name [ptb]')
    #parser.add_argument('--load', type=str, default='MemNN', help='model file to load')
    #parser.add_argument('--save', type=str, default='MemNN', help='path to save model')

    params = parser.parse_args()

    train_data_path = params.data_dir + "/" + params.data_name + "_train.csv"
    test_data_path  = params.data_dir + "/" + params.data_name + "_test.csv"
    max_n_instance, n_question = read_max_n_instance_n_question(train_data_path)
    print max_n_instance, n_question

    train_data = read_content(train_data_path, n_question, params.set_max_seqlen)
    print train_data
    print train_data[train_data.shape[0]-1]
    print train_data.shape
