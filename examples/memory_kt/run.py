import numpy as np
import math
from arena.utils import *

def binaryEntropy(params, target, pred):
    target = target.reshape((-1,))
    loss = 0.
    total = 0
    for i in range(pred.shape[0]):
        next_label = int(target[i])
        #print 'next_label:',next_label
        if next_label > 0.0:
            if next_label > params.n_question:
                next_label -= 1
                correct = 1.0
                next_label -= params.n_question
                #print 'correct', pred[i][next_label]
            else:
                next_label -= 1
                correct = 0.0
                #print 'wrong', pred[i][next_label]
            total += 1
            # Qindex starts from 0, so we need to -1
            loss += - (  correct * np.log( max(1e-10, pred[i][next_label]) )
                        + (1.0 - correct) * np.log( max(1e-10, 1 - pred[i][next_label] )))
    if total == 0:
        print 'total == 0'
        return 0.0
    else:
        #print "total:",total
        return loss / total




def train(net, params, data, label):
    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    np.random.shuffle(data)
    N = int(math.floor(len(data) / params.batch_size))
    data = data.T # Shape: (200,3633)
    cost = 0

    ### Initialization
    init_memory_npy = np.tanh(np.random.normal(size=(params.batch_size, params.memory_size, params.memory_state_dim)))
    #init_h_npy = np.zeros((params.batch_size, params.control_state_dim), dtype=np.float32) + 0.0001
    #init_c_npy = np.zeros((params.batch_size, params.control_state_dim), dtype=np.float32) + 0.0001
    init_h_npy = numpy.tanh(numpy.random.normal(size=(params.batch_size, params.control_state_dim)))
    init_c_npy = numpy.tanh(numpy.random.normal(size=(params.batch_size, params.control_state_dim)))
    init_write_W_r_focus_npy = npy_softmax(numpy.broadcast_to(
                                            numpy.arange(params.memory_size, 0, -1),
                                            (params.batch_size, params.num_writes, params.memory_size)),
                               axis=2)
    init_write_W_u_focus_npy = np.zeros((params.batch_size, params.num_writes, params.memory_size))
    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)

    for idx in xrange(N):
        if params.show: bar.next()
        one_seq = data[: , idx*params.batch_size:(idx+1)*params.batch_size]
        input_x = one_seq[:-1,:] # Shape (seqlen, batch_size)
        target = one_seq[1:,:]

        outputs = net.forward(is_train=True,
                              **{'data': input_x,
                                 'target': target,
                                 'init_memory': init_memory_npy,
                                 'MANN->write_head:init_W_r_focus': init_write_W_r_focus_npy,
                                 'MANN->write_head:init_W_u_focus': init_write_W_u_focus_npy,
                                 'controller->layer0:init_h': init_h_npy,
                                 'controller->layer0:init_c': init_c_npy})
        pred = outputs[0].asnumpy()
        control_state = outputs[1].asnumpy()
        norm_key = outputs[2].asnumpy()
        norm_memory = outputs[3].asnumpy()
        similarity_score = outputs[4].asnumpy()

        #print "Before Updating ......"
        #print "\n"
        #print "norm_key", norm_key.shape, '\n', norm_key[0]
        #print "norm_memory", norm_memory.shape, '\n', norm_memory[0]
        #print "similarity_score", similarity_score.shape, '\n', similarity_score[0]
        #print "control_state", control_state.shape, '\n', control_state[0]
        #print "pred", '\n', pred[0]

        net.backward()
        #print "net.params_grad.items()"
        #for k, v in net.params_grad.items():
        #    print k, "\n", v.asnumpy()
        #    print "                                                                         ---->", \
        #        k, nd.norm(v).asnumpy()
        #print "===========================================================================\n\n\n\n"
        norm_clipping(net.params_grad, params.maxgradnorm)
        optimizer = mx.optimizer.create(name='SGD', learning_rate=params.lr, momentum=params.momentum,
                                        rescale_grad=1.0/ params.batch_size)
        updater = mx.optimizer.get_updater(optimizer)
        #print "net.params_grad.items()"
        #for k, v in net.params_grad.items():
        #    print k, "\n", v.asnumpy()
        #    print "                                                                         ---->", \
        #        k, nd.norm(v).asnumpy()
        #print "===========================================================================\n\n\n\n"
        net.update(updater=updater)
        ### print parameter information
        #print "net.params.items()"
        #for k, v in net.params.items():
        #    print k, "\n", v.asnumpy()
        #    print "                                                                         ---->", \
        #        k, nd.norm(v).asnumpy()
        #print "===========================================================================\n\n"
        #print "net.params_grad.items()"
        #for k, v in net.params_grad.items():
        #    print k, "\n", v.asnumpy()
        #    print "                                                                         ---->",\
        #        k, nd.norm(v).asnumpy()
        #print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n\n\n\n"

        ### get results and compute the loss

        #print "After Updating ......"
        #print "\n"
        #print "norm_key", norm_key.shape, '\n', norm_key[0]
        #print "norm_memory", norm_memory.shape,'\n', norm_memory[0]
        #print "similarity_score", similarity_score.shape, '\n', similarity_score[0]
        #print "control_state", control_state.shape, '\n', control_state[0]
        #print "pred", '\n', pred[0]
        #print "pred.shape", pred.shape # (200L, 111L)
        #print "target.shape", target, target.shape

        avg_loss = binaryEntropy(params, target, pred)
        cost += avg_loss
        #print avg_loss
    if params.show: bar.finish()

    one_epoch_loss = cost / N
    print label, "loss:", one_epoch_loss
    return one_epoch_loss


def test(net, params, data, label):
    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    np.random.shuffle(data)
    N = int(math.floor(len(data) / params.batch_size))
    data = data.T # Shape: (200,3633)
    cost = 0

    ### Initialization
    init_memory_npy = np.tanh(np.random.normal(size=(params.batch_size, params.memory_size, params.memory_state_dim)))
    #init_h_npy = np.zeros((params.batch_size, params.control_state_dim), dtype=np.float32) + 0.0001
    #init_c_npy = np.zeros((params.batch_size, params.control_state_dim), dtype=np.float32) + 0.0001
    init_h_npy = numpy.tanh(numpy.random.normal(size=(params.batch_size, params.control_state_dim)))
    init_c_npy = numpy.tanh(numpy.random.normal(size=(params.batch_size, params.control_state_dim)))
    init_write_W_r_focus_npy = npy_softmax(numpy.broadcast_to(
                                            numpy.arange(params.memory_size, 0, -1),
                                            (params.batch_size, params.num_writes, params.memory_size)),
                               axis=2)
    init_write_W_u_focus_npy = np.zeros((params.batch_size, params.num_writes, params.memory_size))
    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)

    for idx in xrange(N):
        if params.show: bar.next()
        one_seq = data[: , idx*params.batch_size:(idx+1)*params.batch_size]
        input_x = one_seq[:-1,:] # Shape (seqlen, batch_size)
        target = one_seq[1:,:]

        outputs = net.forward(is_train=True,
                              **{'data': input_x,
                                 'target': target,
                                 'init_memory': init_memory_npy,
                                 'MANN->write_head:init_W_r_focus': init_write_W_r_focus_npy,
                                 'MANN->write_head:init_W_u_focus': init_write_W_u_focus_npy,
                                 'controller->layer0:init_h': init_h_npy,
                                 'controller->layer0:init_c': init_c_npy})
        pred = outputs[0].asnumpy()
        avg_loss = binaryEntropy(params, target, pred)
        cost += avg_loss
        #print avg_loss
    if params.show: bar.finish()

    one_epoch_loss = cost / N
    print label, "loss:", one_epoch_loss
    return one_epoch_loss
