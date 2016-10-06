import numpy as np
import math
from arena.utils import *
import mxnet as mx
from arena.helpers.visualization import *
from visualizer import *

def binaryEntropy(params, pred, target):
    loss = target * np.log( np.maximum(1e-10,pred)) + (1.0 - target) * np.log( np.maximum(1e-10, 1.0-pred) )
    return np.average(loss)*(-1.0)

def compute_auc(params, pred, truth ):
    """ compute AUC
    Parameters
        pred : Shape (batch_size*seqlen*N, )
        label : Shape (batch_size*seqlen*N, )
    ------------------------------------------------------
    Returns
        auc : Shape scalar
    """
    pred_truth_array = np.vstack((pred, truth))
    #print pred
    #print truth
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
        if i % 50 == 0 :
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

def train(net, updater, params, q_data, qa_data, label):

    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    N = int(math.floor(len(q_data) / params.batch_size))
    q_data = q_data.T # Shape: (200,3633)
    qa_data = qa_data.T  # Shape: (200,3633)
    # Shuffle the data
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]
    cost = 0

    pred_list = []
    target_list = []

    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)
    for idx in xrange(N):
        if params.show: bar.next()
        q_one_seq = q_data[: , idx*params.batch_size:(idx+1)*params.batch_size]
        input_q = q_one_seq[:,:] # Shape (seqlen, batch_size)
        qa_one_seq = qa_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        input_qa = qa_one_seq[:-1, :]  # Shape (seqlen, batch_size)

        target = qa_one_seq[1:,:]
        target = target.astype(np.int)
        target = (target - 1) / params.n_question
        target = target.astype(np.float) # correct: 1.0; wrong 0.0; padding -1.0

        outputs = net.forward(is_train=True,
                              **{'q_data': input_q,
                                 'qa_data':input_qa,
                                 'target': target})
        pred = outputs[0].asnumpy() #(seqlen * batch_size, 1)
        '''
        control_state = outputs[1].asnumpy()
        read_content = outputs[2].asnumpy()
        key_read_focus = outputs[3].asnumpy()
        value_read_focus = outputs[4].asnumpy()
        '''

        control_state = outputs[4].asnumpy()
        read_content = outputs[5].asnumpy()
        key_read_focus = outputs[6].asnumpy()
        value_read_focus = outputs[7].asnumpy()

        #print "Before Updating ......\n"
        #print "norm_key", norm_key.shape, '\n', norm_key[:,0,:]
        #print "norm_memory", norm_memory.shape, '\n', norm_memory[:,0,:]
        #print "similarity_score", similarity_score.shape, '\n', similarity_score[:,0,:]
        #print "control_state", control_state.shape, '\n', control_state[:,0,:]
        #print "read_content", read_content.shape, '\n', read_content[:,0,:]
        #print "read_focus", read_focus.shape, '\n', read_focus[:, 0, :]
        #print "write_focus", write_focus.shape, '\n', write_focus[:, 0, :]
        #print "pred", '\n', pred[0]

        net.backward()
        norm_clipping(net.params_grad, params.maxgradnorm)
        #print "Before updating,net.params_grad.items()"
        #for k, v in net.params_grad.items():
        #    print k, "\n", v.asnumpy()
        #    print "                                                                         ---->", \
        #        k, nd.norm(v).asnumpy()
        #print "===========================================================================\n\n\n\n"
        net.update(updater=updater)
        #print "After updating, net.params_grad.items()"
        #for k, v in net.params_grad.items():
        #    print k, "\n", v.asnumpy()
        #    print "                                                                         ---->",\
        #        k, nd.norm(v).asnumpy()
        #print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n\n\n\n"
        #print "net.params.items()"
        #for k, v in net.params.items():
        #    print k, "\n", v.asnumpy()
        #    print "                                                                         ---->", \
        #        k, nd.norm(v).asnumpy()
        #print "===========================================================================\n\n"

        ### get results and compute the loss
        #print "After Updating ......"

        target = target.reshape((-1,)) # correct: 1.0; wrong 0.0; padding -1.0
        nopadding_index = np.flatnonzero(target != -1.0)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        avg_loss = binaryEntropy(params, pred_nopadding, target_nopadding)
        cost += avg_loss
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)
        #print avg_loss
    entropy_reg = outputs[1].asnumpy()
    frobenius_reg = outputs[2].asnumpy()
    S = outputs[3].asnumpy()
    if params.show: bar.finish()

    one_epoch_loss = cost / N
    #print label, "loss:", one_epoch_loss
    all_pred = np.concatenate(pred_list,axis=0)
    all_target = np.concatenate(target_list, axis=0)
    accuracy, auc = compute_auc(params, all_pred, all_target)
    return one_epoch_loss, accuracy, auc, entropy_reg, frobenius_reg, S


def test(net, params, q_data, qa_data, label):
    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    #np.random.shuffle(data)
    N = int(math.floor(len(q_data) / params.batch_size))
    q_data = q_data.T  # Shape: (200,3633)
    qa_data = qa_data.T  # Shape: (200,3633)
    cost = 0

    pred_list = []
    target_list = []
    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)


    for idx in xrange(N):
        if params.show: bar.next()
        q_one_seq = q_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
        qa_one_seq = qa_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        input_qa = qa_one_seq[:-1, :]  # Shape (seqlen, batch_size)

        target = qa_one_seq[1:, :]
        target = target.astype(np.int)
        target = (target - 1) / params.n_question
        target = target.astype(np.float)  # correct: 1.0; wrong 0.0; padding -1.0

        outputs = net.forward(is_train=False,
                              **{'q_data': input_q,
                                 'qa_data': input_qa,
                                 'target': target})
        pred = outputs[0].asnumpy()
        '''
        #print "pred.shape",pred.shape, pred # pred.shape (12800L,)#(200L, 64L)
        control_state = outputs[1].asnumpy()
        #print "control_state.shape", control_state.shape, control_state #(200L, 64L, 100L) = seq_len, batch_size, qa_state_dim
        read_content = outputs[2].asnumpy()
        #print "read_content.shape", read_content.shape, read_content #(200L, 64L, 100L) = seq_len, batch_size, qa_state_dim
        key_read_focus = outputs[3].asnumpy()
        #print "key_read_focus.shape", key_read_focus.shape, key_read_focus #(200L, 64L, 100L)= seq_len, batch_size, memory_size
        value_read_focus = outputs[4].asnumpy()
        #print "value_read_focus.shape", value_read_focus.shape, value_read_focus #e (200L, 64L, 100L)
        '''
        control_state = outputs[4].asnumpy()
        read_content = outputs[5].asnumpy()
        key_read_focus = outputs[6].asnumpy()
        value_read_focus = outputs[7].asnumpy()
        if params.vis:
            # read_focus -- Shape ( sequence length, batch size, memory size )
            # write_focus -- Shape ( sequence length, batch size, memory size )
            # control_state -- Shape (sequence length, batch size, control state dim)
            # read_content -- Shape (sequence length, batch size, memory state dim)
            #print "target",target.shape,target
            #vis_kt_one(params, idx, pred, target, control_state, read_content, key_read_focus, value_read_focus)

            # save data as a whole matrix
            num_of_batch = (params.n_question-1) / params.seqlen + 1
            remainer = (params.n_question-1) % params.seqlen
            print "remainer",remainer
            print "num_of_batch", num_of_batch
            all_key_read_focus = np.zeros((params.n_question-1, params.memory_size))
            for i in range(num_of_batch):
                if i != (num_of_batch-1):
                    all_key_read_focus[ i*params.seqlen:(i+1)*params.seqlen, :] = key_read_focus[:,i,:]
                elif i == (num_of_batch-1):
                    all_key_read_focus[i*params.seqlen: , : ] = key_read_focus[ :remainer,i,:]
            print "all_key_read_focus", all_key_read_focus
            np.save(os.path.join('result',params.load,'all_key_read_focus'), all_key_read_focus)

            vis_weight(params, idx, target, key_read_focus, value_read_focus)
            break

        target = target.reshape((-1,))  # correct: 1.0; wrong 0.0; padding -1.0
        nopadding_index = np.flatnonzero(target != -1.0)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]
        avg_loss = binaryEntropy(params, pred_nopadding, target_nopadding)
        cost += avg_loss
        #print avg_loss
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)
    if params.show: bar.finish()

    one_epoch_loss = cost / N
    #print label, "loss:", one_epoch_loss
    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    accuracy, auc = compute_auc(params, all_pred, all_target)
    return one_epoch_loss, accuracy, auc
