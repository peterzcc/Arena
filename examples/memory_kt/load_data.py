import numpy as np


class DATA(object):
    def __init__(self, n_question, seqlen, separate_char, name="data"):
        # In the ASSISTment dataset:
        # param: n_queation = 111
        #        seqlen = 200
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen

    ### data format
    ### 15
    ### 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54,
    ### 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,
    def load_data(self, path):
        f_data = open(path , 'r')
        data = []
        for lineID, line in enumerate(f_data):
            line = line.strip( )
            # lineID starts from 0
            if lineID % 3 == 1:
                Q = line.split(self.separate_char)
                if len( Q[len(Q)-1] ) == 0:
                    Q = Q[:-1]
                #print len(Q)
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len( A[len(A)-1] ) == 0:
                    A = A[:-1]
                #print len(A),A

                # start split the data
                n_split = 1
                #print 'len(Q):',len(Q)
                if len(Q) > self.seqlen:
                    n_split = len(Q) / self.seqlen
                    if len(Q) % self.seqlen:
                        n_split = len(Q) / self.seqlen + 1
                #print 'n_split:',n_split
                for k in range(n_split):
                    instance = []
                    if k == n_split - 1:
                        endINdex  = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(Q[i]) > 0 :
                            # int(A[i]) is in {0,1}
                            Xindex = int(Q[i]) + int(A[i]) * self.n_question
                            instance.append(Xindex)
                        else:
                            print Q[i]
                    #print 'instance:-->', len(instance),instance
                    data.append(instance)
        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used

        # data size = (2 * seqlen + 2,  batch_size, data_dim)
        ### convert data into ndarrays for better speed during training
        dataArray = np.zeros((len(data), self.seqlen))
        for j in range(len(data)):
            dat = data[j]
            dataArray[j, :len(dat)] = dat
        # dataArray: [ array([[],[],..])] Shape: (3633, 200)
        return dataArray



