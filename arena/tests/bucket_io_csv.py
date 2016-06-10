# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

# The interface of a data iter that works for bucketing
#
# DataIter
#   - default_bucket_key: the bucket key for the default symbol.
#
# DataBatch
#   - provide_data: same as DataIter, but specific to this batch
#   - provide_label: same as DataIter, but specific to this batch
#   - bucket_key: the key for the bucket that should be used for this batch

def read_data_information(path,separate_char):
    f_data = open(path,'r')
    max_n_instance = 0
    max_questionID = 0
    for lineID, line in enumerate(f_data):
        line = line.strip( )
        if lineID % 3 == 0:
            n_instance = int(line)
            if n_instance > max_n_instance:
                max_n_instance = n_instance
        elif lineID % 3 == 1:
            Q = line.split(separate_char)
            for i,q in enumerate(Q):
                if len(q) > 0:
                    questionID = int(q)
                    if questionID > max_questionID:
                        max_questionID = questionID
    f_data.close()
    return max_questionID

def default_read_content(path, separate_char, n_question, max_n_question):
    f_data = open(path , 'r')
    data = []
    for lineID, line in enumerate(f_data):
        line = line.strip( )
        # lineID starts from 0
        if lineID % 3 == 1:
            Q = line.split(separate_char)
            if len( Q[len(Q)-1] ) == 0:
                Q = Q[:-1]
            #print len(Q),Q
        elif lineID % 3 == 2:
            A = line.split(separate_char)
            if len( A[len(A)-1] ) == 0:
                A = A[:-1]
            #print len(A),A

            # start split the data
            n_split = 1
            #print 'len(Q):',len(Q)
            if len(Q) > max_n_question:
                n_split = len(Q) / max_n_question
                if len(Q) % max_n_question:
                    n_split = len(Q) / max_n_question + 1
            #print 'n_split:',n_split
            for k in range(n_split):
                instance = []
                if k == n_split - 1:
                    endINdex  = len(A)
                else:
                    endINdex = (k+1) * max_n_question
                for i in range(k*max_n_question, endINdex):
                    if len(Q[i]) > 0 :
                        # int(A[i]) is in {0,1}
                        Xindex = int(Q[i]) + int(A[i]) * n_question
                        instance.append(Xindex)
                    else:
                        print Q[i]
                #print 'instance:-->'
                #print len(instance),instance
                data.append(instance)
    f_data.close()
    #print len(data)
    #data: [[],[],[],...] --> max_n_question is used
    return data


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class BucketQuestionIter(mx.io.DataIter):
    #data_train = BucketQuestionIter(path = os.path.join(data_dir, "builder_train.csv"),
    #                                buckets = buckets, batch_size = batch_size, init_states = init_states)
    def __init__(self, path, buckets, max_n_question, batch_size, init_states,
                 data_name='data', label_name='label',
                 separate_char=',', read_content=None):
        super(BucketQuestionIter, self).__init__()

        # read data the max_questionID and max_sequence_length
        n_question = read_data_information(path,separate_char)

        if read_content == None:
            # def default_read_content(path, seperate_char, n_question, max_n_question):
            self.read_data = default_read_content
        else:
            self.read_data = read_content
        data = self.read_data(path, separate_char, n_question, max_n_question)

        #print 'data', data
        #if len(buckets) == 0:
        #    buckets = default_gen_buckets(sentences, batch_size, vocab)

        self.data_name = data_name
        self.label_name = label_name

        buckets.sort()
        self.buckets = buckets
        self.data = [[] for _ in buckets]

        # pre-allocate with the largest bucket for better memory sharing
        self.default_bucket_key = max(buckets)

        # put data into the corresponding bucket by sequence length
        for dat in data:
            if len(dat) == 0:
                continue
            for i, bkt in enumerate(buckets):
                # we just ignore the sentence it is longer than the maximum
                # bucket size here
                if bkt >= len(dat):
                    self.data[i].append(dat)
                    break

        # convert data into ndarrays for better speed during training
        dataArray = [np.zeros((len(x), buckets[i])) for i, x in enumerate(self.data)]
        #print "dataArray:",dataArray
        for i_bucket in range(len(self.buckets)):
            for j in range(len(self.data[i_bucket])):
                dat = self.data[i_bucket][j]
                dataArray[i_bucket][j, :len(dat)] = dat
        # self.data: [ array([[],[],..]),array([[],[],..]),... ]
        self.data = dataArray
        #print self.data

        # Get the size of each bucket, so that we could sample
        # uniformly from the bucket
        bucket_sizes = [len(x) for x in self.data]

        print("Summary of " + path + " ==================")
        for bkt, size in zip(buckets, bucket_sizes):
            print("bucket of len %3d : %d samples" % (bkt, size))

        #for i,dat in enumerate(self.data):
        #    print dat.shape
        #print self.data

        self.batch_size = batch_size
        self.make_data_iter_plan()

        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('data', (batch_size, self.default_bucket_key))] + init_states
        self.provide_label = [('softmax_label', (self.batch_size, self.default_bucket_key))]

    def make_data_iter_plan(self):
        # "make a random data iteration plan"
        # truncate each bucket into multiple of batch-size
        bucket_n_batches = []
        for i_bucket in range(len(self.data)):
            bucket_n_batches.append( len(self.data[i_bucket]) / self.batch_size )
            self.data[i_bucket] = self.data[i_bucket][:int(bucket_n_batches[i_bucket]*self.batch_size)]

        # bucket_plan : array([0,0,0,1,1,2,2,2,3,3,3]) if bucket_n_batches=[3,2,3,3]
        bucket_plan = np.hstack([np.zeros(n, int)+i for i, n in enumerate(bucket_n_batches)])
        print 'bucket_plan',bucket_plan
        np.random.shuffle(bucket_plan)

        # [array([1,2,0]), array([3,5,2,1,0,4])] --> shuffle the data within each bucket
        bucket_idx_all = [np.random.permutation(len(x)) for x in self.data]

        self.bucket_plan = bucket_plan
        self.bucket_idx_all = bucket_idx_all
        self.bucket_curr_idx = [0 for x in self.data]

        self.data_buffer = []
        self.label_buffer = []
        for i_bucket in range(len(self.data)):
            # In each bucket choose one batch data
            data = np.zeros((self.batch_size, self.buckets[i_bucket]))   # batch_size * each bucket length
            label = np.zeros((self.batch_size, self.buckets[i_bucket]))  # batch_size * each bucket length
            self.data_buffer.append(data)
            self.label_buffer.append(label)
        #print "self.data_buffer:",self.data_buffer      # all zeros
        #print "self.label_buffer:",self.label_buffer    # all zeros
        #print "self.bucket_plan:",self.bucket_plan      # shuffled data 0,1,2,3 if there are 4 buckets

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]

        for i_bucket in self.bucket_plan:
            # initialization
            data = self.data_buffer[i_bucket]
            label = self.label_buffer[i_bucket]
            # choose index
            i_idx = self.bucket_curr_idx[i_bucket]
            idx = self.bucket_idx_all[i_bucket][i_idx:i_idx+self.batch_size]
            self.bucket_curr_idx[i_bucket] += self.batch_size

            data[:] = self.data[i_bucket][idx]
            # make the next input as the output label
            label[:, :-1] = data[:, 1:]
            label[:, -1] = 0

            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['softmax_label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                     self.buckets[i_bucket])
            yield data_batch

    def reset(self):
        self.bucket_curr_idx = [0 for x in self.data]
