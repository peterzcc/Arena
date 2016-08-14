import numpy as np
import math
from arena.utils import *
import mxnet as mx
from arena.helpers.visualization import *


def binaryEntropy(params, pred, target):
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


def compute_auc(params, all_pred, label ):
    """ compute AUC
    Parameters
        pred : Shape (batch_size*seqlen*N, n_question)
        label : Shape (batch_size*seqlen*N, )
    ------------------------------------------------------
    Returns
        auc : Shape scalar
    """
    label = label.astype(np.int)
    zero_index = np.flatnonzero(label == 0)
    non_zero_index = np.flatnonzero(label)
    next_label = (label - 1) % params.n_question # Shape (batch_size*seqlen*N, )
    truth = (label - 1) / params.n_question # Shape (batch_size*seqlen*N, )
    next_label[zero_index] = 0
    truth[zero_index] = 0
    next_label = next_label.tolist()
    prediction = all_pred[np.arange(len(next_label)), next_label] # Shape (batch_size*seqlen*N, )
    pre = prediction[non_zero_index]
    tru = truth[non_zero_index]
    pred_truth_array = np.vstack((pre,tru))
    pred_truth_array = pred_truth_array.T
    #print "pred_truth_array.shape", pred_truth_array.shape

    #print "\n\n\n\nStart computing AUC ......"
    # sort the array according to the the first column
    pred_truth_array = pred_truth_array[pred_truth_array[:,0].argsort()[::-1]]
    #print 'pred_truth_array', pred_truth_array.shape, pred_truth_array
    #f_save = open('pred_truth_array','wb')
    #np.save(f_save, pred_truth_array)
    #f_save.close()
    # start computing AUC
    allPredictions = pred_truth_array.shape[0]
    total_positives = np.sum(pred_truth_array[:,1])
    total_negatives = allPredictions - total_positives
    #print 'total_positives', total_positives
    #print 'total_negatives', total_negatives

    true_positives = 0
    false_positives = 0
    correct = 0
    auc = 0.0
    # pred_truth_list[i,0] --> predicted value; pred_truth_list[i,1] --> truth value
    lastTpr = 0.0
    lastFpr = 0.0
    for i in range(allPredictions):
        truth = int(pred_truth_array[i,1]) # truth in {0,1}
        if truth == 1:
            true_positives += 1
        else:
            #print "false_positives:",false_positives
            false_positives += 1
        fpr = float(false_positives) / float(total_negatives)
        tpr = float(true_positives) / float(total_positives)
        # using trapezoid method to compute auc
        if i % 500 == 0 :
            #print i
            trapezoid = (tpr + lastTpr) * (fpr - lastFpr) * 0.5
            #print "trapzoid:",trapezoid
            #print "auc:",auc
            auc += trapezoid
            lastTpr = tpr
            lastFpr = fpr
        # computing accuracy
        if pred_truth_array[i,0] > 0.5 :
            guess = 1
        else:
            guess = 0
        if guess == truth:
            correct += 1
    accuracy = float(correct) /float(allPredictions)
    return accuracy, auc


def onehot_encoding(n_question, seqlen, label):
    one_hot = np.zeros((seqlen, n_question))
    label = label.astype(np.int)
    zero_index = np.flatnonzero(label == 0)
    non_zero_index = np.flatnonzero(label)
    next_label = (label - 1) % n_question  # Shape (batch_size*seqlen*N, )
    truth = (label - 1) / n_question  # Shape (batch_size*seqlen*N, )
    next_label[zero_index] = 0
    truth[zero_index] = 0
    next_label = next_label.tolist()
    one_hot[np.arange(len(next_label)), next_label] = truth[np.arange(len(next_label))]
    return one_hot

def vis_matrix(params, pred, target):
    # :Parameter pred : Shape (seqlen*batch_size , n_question)
    vis_pred_all = pred.reshape( (params.seqlen, params.batch_size, params.n_question) )
    vis_pred_all = vis_pred_all[:,0,:] # Shape ( seqlen, n_question ) when vis using .T
    target = target[:,0]
    #print target
    #print "target.shape", target.shape
    vis_target_one_hot = np.zeros((params.seqlen, params.n_question))
    vis_pred_one_hot = np.zeros((params.seqlen, params.n_question))
    target = target.astype(np.int)
    zero_index = np.flatnonzero(target == 0)
    #print "zero_index", zero_index
    non_zero_index = np.flatnonzero(target)
    #print "non_zero_index", non_zero_index
    next_label = (target - 1) % params.n_question  # Shape (batch_size*seqlen*N, )
    #print "next_label", next_label
    truth = (target - 1) / params.n_question  # Shape (batch_size*seqlen*N, )
    #print "truth",truth
    next_label[zero_index] = 0
    #print "next_label", next_label
    ### correct 1 , wrong 0, no answer -1
    #truth[zero_index] = -1
    ### correct 1 , wrong -1, no answer 0
    truth_wrong = np.flatnonzero(truth == 0)
    truth[truth_wrong] = -1
    truth[zero_index] = 0
    #print "truth", truth
    next_label = next_label.tolist()
    vis_target_one_hot[np.arange(len(next_label)), next_label] = truth[np.arange(len(next_label))]
    vis_pred_one_hot[np.arange(len(next_label)), next_label] = vis_pred_all[np.arange(len(next_label)), next_label]
    vis_pred_one_hot[zero_index,0] = 0.0
    #print "vis_target_one_hot", vis_target_one_hot
    #print "vis_pred_one_hot", vis_pred_one_hot
    #result_pred = vis_pred_one_hot[np.arange(len(next_label)),next_label]
    #result_target = vis_target_one_hot[np.arange(len(next_label)),next_label]
    #print "result_pred", result_pred
    #print "result_target", result_target
    return vis_pred_all, vis_pred_one_hot, vis_target_one_hot

def train(net, params, data, label):

    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    np.random.shuffle(data)
    N = int(math.floor(len(data) / params.batch_size))
    data = data.T # Shape: (200,3633)
    cost = 0

    ### Initialization
    init_memory_npy = np.tanh(np.random.normal(size=(params.batch_size, params.memory_size, params.memory_state_dim)))
    init_h_npy = np.zeros((params.batch_size, params.control_state_dim), dtype=np.float32) + 0.0001
    init_c_npy = np.zeros((params.batch_size, params.control_state_dim), dtype=np.float32) + 0.0001
    #init_h_npy = numpy.tanh(np.random.normal(size=(params.batch_size, params.control_state_dim)))
    #init_c_npy = numpy.tanh(np.random.normal(size=(params.batch_size, params.control_state_dim)))
    init_write_W_r_focus_npy = npy_softmax(np.broadcast_to(
                                           np.arange(params.memory_size, 0, -1),
                                           (params.batch_size, params.num_writes, params.memory_size)),
                               axis=2)
    init_write_W_u_focus_npy = np.zeros((params.batch_size, params.num_writes, params.memory_size))
    pred_list = []
    target_list = []

    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)
    optimizer = mx.optimizer.create(name='SGD', learning_rate=params.lr, momentum=params.momentum,
                                    rescale_grad=1.0 / params.batch_size)
    updater = mx.optimizer.get_updater(optimizer)
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

        read_content = outputs[5].asnumpy()
        print "read_content.shape", read_content.shape
        read_focus = outputs[6].asnumpy()
        print "read_focus.shape", read_focus.shape
        print read_focus[:, 0, :]
        print read_focus[:, 0, :].max(axis=1), read_focus[:, 0, :].max(axis=1).shape
        write_focus = outputs[7].asnumpy()
        print "write_focus.shape", write_focus.shape
        print write_focus[:, 0, :]
        print write_focus[:, 0, :].max(axis=1), write_focus[:, 0, :].max(axis=1).shape

        if params.vis:

            # print target
            vis_pred_all, vis_pred_one_hot, vis_target_one_hot = vis_matrix(params, pred, target)
            # print vis_target\
            #print vis_pred[:, 0, :].T.shape
            #print vis_target[:, 0, :].T.shape
            #print vis_pred
            CV2Vis.display(data=vis_pred_all.T, win_name="prediction_all", delay=1)
            CV2Vis.display(data=vis_pred_one_hot.T, win_name="prediction_one_hot", delay=1)
            CV2Vis.display(data=vis_target_one_hot.T, win_name="target_one_hot", delay=1)
            CV2Vis.display(data=read_content[:,0,:].T, win_name="read_content", delay=1)
            CV2Vis.display(data=read_focus[:, 0, :].T, win_name="read_focus", delay=1)
            CV2Vis.display(data=write_focus[:, 0, :].T, win_name="write_focus", delay=1)
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
            #print k, "\n", v.asnumpy()
        #    print k, '\t\t', nd.norm(v).asnumpy()
        #print "===========================================================================\n\n\n\n"
        norm_clipping(net.params_grad, params.maxgradnorm)
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

        target = target.reshape((-1,))
        avg_loss = binaryEntropy(params, pred, target)
        cost += avg_loss
        pred_list.append(pred)
        target_list.append(target)
        #print avg_loss
    if params.show: bar.finish()

    one_epoch_loss = cost / N
    #print label, "loss:", one_epoch_loss
    all_pred = np.concatenate(pred_list,axis=0)
    all_target = np.concatenate(target_list, axis=0)
    accuracy, auc = compute_auc(params, all_pred, all_target)
    if params.vis:
        vis_pred = outputs[0].reshape((params.seqlen, params.batch_size, params.n_question)).asnumpy()
        # print target
        vis_target = onehot_encoding(params.n_question, params.seqlen * params.batch_size, target).reshape(
            (params.seqlen, params.batch_size, params.n_question))
        # print vis_target\
        print vis_pred[:, 0, :].T.shape
        print vis_target[:, 0, :].T.shape
        CV2Vis.display(data=vis_pred[:, 0, :].T, win_name="prediction")
        CV2Vis.display(data=vis_target[:, 0, :].T, win_name="target")

    return one_epoch_loss, accuracy, auc


def test(net, params, data, label):
    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    np.random.shuffle(data)
    N = int(math.floor(len(data) / params.batch_size))
    data = data.T # Shape: (200,3633)
    cost = 0

    ### Initialization
    init_memory_npy = np.tanh(np.random.normal(size=(params.batch_size, params.memory_size, params.memory_state_dim)))
    init_h_npy = np.zeros((params.batch_size, params.control_state_dim), dtype=np.float32) + 0.0001
    init_c_npy = np.zeros((params.batch_size, params.control_state_dim), dtype=np.float32) + 0.0001
    #init_h_npy = numpy.tanh(numpy.random.normal(size=(params.batch_size, params.control_state_dim)))
    #init_c_npy = numpy.tanh(numpy.random.normal(size=(params.batch_size, params.control_state_dim)))
    init_write_W_r_focus_npy = npy_softmax(np.broadcast_to(
                                            np.arange(params.memory_size, 0, -1),
                                            (params.batch_size, params.num_writes, params.memory_size)),
                               axis=2)
    init_write_W_u_focus_npy = np.zeros((params.batch_size, params.num_writes, params.memory_size))
    pred_list = []
    target_list = []
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
        target = target.reshape((-1,))
        avg_loss = binaryEntropy(params, pred, target)
        cost += avg_loss
        #print avg_loss
        pred_list.append(pred)
        target_list.append(target)
    if params.show: bar.finish()

    one_epoch_loss = cost / N
    #print label, "loss:", one_epoch_loss
    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    accuracy, auc = compute_auc(params, all_pred, all_target)
    return one_epoch_loss, accuracy, auc
