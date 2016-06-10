# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python")))
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rnn")))
import numpy as np
import mxnet as mx
from arena.advanced.lstm import lstm_unroll
from bucket_io_csv import BucketQuestionIter
import os
import os.path
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ASSISTment'))

#def Perplexity(label, pred):
#    label = label.T.reshape((-1,))
#    loss = 0.
    #print label
#    for i in range(pred.shape[0]):
#        next_label = int(label[i])
#        if next_label > 111:
#            next_label -= 111
#        next_label -= 1
#        loss += -np.log(max(1e-10, pred[i][next_label]))
#    return np.exp(loss / label.size)

# label = batch_size * bucket _size
# pred = (batch_size*bucket _size) * output_size
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
    data_test = BucketQuestionIter(path = os.path.join(data_dir, "builder_test.csv"),
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



