# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
import os
sys.path.insert(0, "/Users/jenny/Documents/mxnet/python")
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rnn")))
import numpy as np
import mxnet as mx
from arena.advanced.lstm import lstm_unroll
from bucket_io_csv import BucketQuestionIter
import os
import os.path
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ASSISTment'))


def binaryEntropy(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        next_label = int(label[i]) - 1
        print 'next_label:',next_label
        loss += -np.log(max(1e-10, pred[i][next_label]))
    return loss / label.size


def binaryEntropy2(label, pred):
    n_q = 111
    label = label.T.reshape((-1,))
    loss = 0.
    total = 0
    for i in range(pred.shape[0]):
        next_label = int(label[i])
        #print 'next_label:',next_label
        if next_label > 0.0:
            if next_label > n_q:
                next_label -= 1
                correct = 1.0
                next_label -= n_q
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


if __name__ == '__main__':
    #########################   model parameter setting         #########################
    batch_size = 32
    max_length = 200
    n_q = 111
    buckets = [50, 100, 150, max_length]
    #buckets = [32]
    num_hidden = 100
    num_embed = 100
    num_lstm_layer = 1
    #########################   training parameter setting      #########################
    #num_epoch = 25
    num_epoch = 20
    learning_rate = 0.1
    momentum = 0.9
    #contexts = [mx.context.gpu(i) for i in range(1)]
    contexts = [mx.context.cpu()]
    #########################   model initialization            #########################
    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = BucketQuestionIter(path = os.path.join(data_dir, "builder_train.csv"),
                                    buckets = buckets, max_n_question = max_length,
                                    batch_size = batch_size, init_states = init_states)
    data_test  = BucketQuestionIter(path = os.path.join(data_dir, "builder_test.csv"),
                                    buckets = buckets, max_n_question = max_length,
                                    batch_size = batch_size, init_states = init_states)

    #########################   model parameter setting         #########################
    state_names = [x[0] for x in init_states]

    print "Start training ...... ", "Learning Rate = ", learning_rate,"Momentum = ",momentum
    def sym_gen(seq_len):
        # def lstm_unroll(num_lstm_layer, seq_len, input_size,
        #        num_hidden, num_embed, num_label, dropout=0.):
        sym = lstm_unroll(num_lstm_layer = num_lstm_layer, seq_len = seq_len, input_size = n_q * 2, # Here maybe the problem should go in deep
                          num_hidden = num_hidden, num_embed = num_embed, num_label = n_q)
        data_names = ['data'] + state_names
        label_names = ['softmax_label']
        return (sym, data_names, label_names)

    if len(buckets) == 1:
        mod = mx.mod.Module(*sym_gen(buckets[0]), context=contexts)
    else:
        mod = mx.mod.BucketingModule(sym_gen, default_bucket_key=data_train.default_bucket_key, context=contexts)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    #for nbatch, data_batch in enumerate(data_train):
    #    print data_batch.label
    mod.fit(train_data = data_train, eval_data = data_test, num_epoch = num_epoch,
            eval_metric = mx.metric.np(binaryEntropy2),
            batch_end_callback = mx.callback.Speedometer(batch_size, 50),
            #initializer = mx.init.Normal(sigma=0.01),
            optimizer = 'sgd',
            optimizer_params = {'learning_rate':learning_rate, 'momentum':momentum , 'wd': 0.00001})

    # compute AUC
    pred_truth_array = np.zeros((1, 2))
    for pred, i_batch, batch in mod.iter_predict(eval_data=data_test):
        # pred: list of only one object: [ mxnet.ndarray.NDArray object ]
        # i_batch is a integer
        # batch is the data batch from the data iterator

        # print "pred[0].asnumpy().shape", pred[0].asnumpy().shape # 'list' object has no attribute 'asnumpy'
        pred = pred[0].asnumpy()
        # print "pred.shape:", pred.shape # (batch_size * bucket_length, 111L)
        # print "batch.label[0].asnumpy().shape:",batch.label[0].asnumpy().shape # (batch_size, bucket_length)
        label = batch.label[0].asnumpy().T.reshape((-1,)).astype(np.int)
        # print "label:",label
        # print "label.shape:",label.shape # label.shape: (1000L,) --> (batch_size*bucket_length)

        zero_index = np.flatnonzero(label == 0)
        non_zero_index = np.flatnonzero(label)
        # print "zero_index",zero_index.shape,zero_index
        # print "non_zero_index", non_zero_index.shape, non_zero_index
        all_prediction_num = label.size - zero_index.size
        # print "all_prediction_num", all_prediction_num
        next_label = (label - 1) % n_q
        truth = (label - 1) / n_q
        next_label[zero_index] = 0
        truth[zero_index] = 0
        next_label = next_label.tolist()
        # print 'next_label', next_label
        # print 'truth',truth
        prediction = pred[np.arange(len(next_label)), next_label]
        # print "prediction",prediction
        tru = truth[non_zero_index].tolist()
        pre = prediction[non_zero_index].tolist()
        # print "tru", len(tru), tru
        # print "pre", len(pre), pre
        one_batch_pre_truth = []
        one_batch_pre_truth.append(pre)
        one_batch_pre_truth.append(tru)
        one_batch_pre_truth = np.asarray(one_batch_pre_truth).T
        # print 'one_batch_pre_truth',one_batch_pre_truth.shape,one_batch_pre_truth
        pred_truth_array = np.concatenate((pred_truth_array, one_batch_pre_truth), axis=0)
        # print "\n\n"

    # Get all prediction results
    pred_truth_array = pred_truth_array[1:, :]
    # print 'pre_truth',pred_truth_array.shape, pred_truth_array
    print "\n\n\n\nStart computing AUC ......"
    # sort the array according to the the first column
    pred_truth_array = pred_truth_array[pred_truth_array[:, 0].argsort()[::-1]]
    print 'pred_truth_array', pred_truth_array.shape, pred_truth_array
    f_save = open('pred_truth_array', 'wb')
    np.save(f_save, pred_truth_array)
    f_save.close()
    # start computing AUC
    allPredictions = pred_truth_array.shape[0]
    total_positives = np.sum(pred_truth_array[:, 1])
    total_negatives = allPredictions - total_positives
    print 'total_positives', total_positives
    print 'total_negatives', total_negatives
    true_positives = 0
    false_positives = 0
    correct = 0
    auc = 0.0
    # pred_truth_list[i,0] --> predicted value; pred_truth_list[i,1] --> truth value
    lastTpr = 0.0
    lastFpr = 0.0
    for i in range(allPredictions):
        truth = int(pred_truth_array[i, 1])  # truth in {0,1}
        if truth == 1:
            true_positives += 1
        else:
            # print "false_positives:",false_positives
            false_positives += 1
        fpr = float(false_positives) / float(total_negatives)
        tpr = float(true_positives) / float(total_positives)
        # using trapezoid method to compute auc

        if i % 500 == 0:
            # print i
            trapezoid = (tpr + lastTpr) * (fpr - lastFpr) * 0.5
            # print "trapzoid:",trapezoid
            # print "auc:",auc
            auc += trapezoid
            lastTpr = tpr
            lastFpr = fpr
        # computing accuracy
        if pred_truth_array[i, 0] > 0.5:
            guess = 1
        else:
            guess = 0
        if guess == truth:
            correct += 1

    accuracy = float(correct) / float(allPredictions)

    print "======> accuracy of testing is ", accuracy, "auc of testing is ", auc

