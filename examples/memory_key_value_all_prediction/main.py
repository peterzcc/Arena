import logging
import argparse
from load_data import DATA
from arena import Base
from model import MODEL
from run import train
from run import test
from arena.helpers.visualization import *


root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

class KVMNInitializer(mx.initializer.Normal):
    def _init_default(self, name, arr):
        if "init_memory" in name:
            arr[:] = numpy.random.normal(loc=0.1, size=arr.shape)
        else:
            assert False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test KVMN.')
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--max_iter', type=int, default=100, help='number of iterations')
    parser.add_argument('--num_heads', type=int, default=1, help='number of tensors')

    parser.add_argument('--test', type=bool, default=False, help='enable testing')
    parser.add_argument('--show', type=bool, default=True, help='print progress')
    parser.add_argument('--vis', type=bool, default=False, help='visualize weights and results')

    dataset = "KDDal0506" #  assistment2009 / assistment2015 / KDDal0506 / STATICS
    if dataset == "assistment2009":
        ### assistment2009
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--q_state_dim', type=int, default=20, help='hidden states of question')
        parser.add_argument('--qa_embed_dim', type=int, default=100, help='answer and question embedding dimensions')
        parser.add_argument('--qa_state_dim', type=int, default=50, help='hidden states of answer and question')

        parser.add_argument('--memory_size', type=int, default=50, help='memory size')
        parser.add_argument('--memory_key_state_dim', type=int, default=20, help='the key part of memory state dimension')
        parser.add_argument('--memory_value_state_dim', type=int, default=50, help='the value part of memory state dimension')

        parser.add_argument('--init_std', type=float, default=0.05, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.1, help='initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')

        parser.add_argument('--n_question', type=int, default=123, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='data/assistment2009', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2009', help='data set name')
        parser.add_argument('--load', type=str, default='assistment2009', help='model file to load')
        parser.add_argument('--save', type=str, default='assistment2009', help='path to save model')

    elif dataset == "assistment2015":
        ### assistment2015
        parser.add_argument('--batch_size', type=int, default=50, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--q_state_dim', type=int, default=25, help='hidden states of question')
        parser.add_argument('--qa_embed_dim', type=int, default=100, help='answer and question embedding dimensions')
        parser.add_argument('--qa_state_dim', type=int, default=50, help='hidden states of answer and question')

        parser.add_argument('--memory_size', type=int, default=20, help='memory size')
        parser.add_argument('--memory_key_state_dim', type=int, default=25, help='the key part of memory state dimension')
        parser.add_argument('--memory_value_state_dim', type=int, default=50, help='the value part of memory state dimension')

        parser.add_argument('--init_std', type=float, default=0.05, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')

        parser.add_argument('--n_question', type=int, default=100, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='data/assistment2015', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2015', help='data set name')
        parser.add_argument('--load', type=str, default='assistment2015', help='model file to load')
        parser.add_argument('--save', type=str, default='assistment2015', help='path to save model')

    elif dataset == "KDDal0506":
        ### KDDal0506
        parser.add_argument('--batch_size', type=int, default=50, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--q_state_dim', type=int, default=25, help='hidden states of question')
        parser.add_argument('--qa_embed_dim', type=int, default=100, help='answer and question embedding dimensions')
        parser.add_argument('--qa_state_dim', type=int, default=50, help='hidden states of answer and question')

        parser.add_argument('--memory_size', type=int, default=50, help='memory size')
        parser.add_argument('--memory_key_state_dim', type=int, default=25, help='the key part of memory state dimension')
        parser.add_argument('--memory_value_state_dim', type=int, default=50, help='the value part of memory state dimension')

        parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.1, help='initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')

        parser.add_argument('--n_question', type=int, default=436, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='data/KDDal0506', help='data directory')
        parser.add_argument('--data_name', type=str, default='KDDal0506', help='data set name')
        parser.add_argument('--load', type=str, default='KDDal0506', help='model file to load')
        parser.add_argument('--save', type=str, default='KDDal0506', help='path to save model')
    elif dataset == "STATICS":
        ### STATICS
        parser.add_argument('--batch_size', type=int, default=10, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=50, help='question embedding dimensions')
        parser.add_argument('--q_state_dim', type=int, default=25, help='hidden states of question')
        parser.add_argument('--qa_embed_dim', type=int, default=100, help='answer and question embedding dimensions')
        parser.add_argument('--qa_state_dim', type=int, default=50, help='hidden states of answer and question')

        parser.add_argument('--memory_size', type=int, default=50, help='memory size')
        parser.add_argument('--memory_key_state_dim', type=int, default=25, help='the key part of memory state dimension')
        parser.add_argument('--memory_value_state_dim', type=int, default=50, help='the value part of memory state dimension')

        parser.add_argument('--init_std', type=float, default=0.01, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')

        parser.add_argument('--n_question', type=int, default=1223, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='data/STATICS', help='data directory')
        parser.add_argument('--data_name', type=str, default='STATICS', help='data set name')
        parser.add_argument('--load', type=str, default='STATICS', help='model file to load')
        parser.add_argument('--save', type=str, default='STATICS', help='path to save model')



    params = parser.parse_args()
    print params
    params.lr = params.init_lr
    ### ================================== choose ctx ==================================
    if params.gpus == None:
        ctx = mx.cpu()
        print "Training with cpu ..."
    else:
        ctx = mx.gpu(int(params.gpus))
        print "Training with gpu(" + params.gpus + ") ..."
    ### ================================== model initialization ==================================
    g_model = MODEL(n_question=params.n_question,
                    seqlen=params.seqlen,
                    batch_size=params.batch_size,
                    q_embed_dim=params.q_embed_dim,
                    q_state_dim=params.q_state_dim,
                    qa_embed_dim=params.qa_embed_dim,
                    qa_state_dim=params.qa_state_dim,
                    memory_size=params.memory_size,
                    memory_key_state_dim=params.memory_key_state_dim,
                    memory_value_state_dim=params.memory_value_state_dim,
                    num_heads=params.num_heads)
    data_shapes = {'q_data': (params.seqlen+1, params.batch_size),
                   'qa_data': (params.seqlen, params.batch_size),
                   'target': (params.seqlen, params.batch_size),
                   'init_memory_key': (params.memory_size, params.memory_key_state_dim),
                   'init_memory_value': (params.memory_size, params.memory_value_state_dim)}
    learn_init_keys = ['init_memory_key',
                       'init_memory_value']
    initializer = KVMNInitializer(sigma=params.init_std)
    net = Base(sym_gen=g_model.sym_gen(),
               data_shapes=data_shapes,
               learn_init_keys=learn_init_keys,
               initializer=initializer,
               ctx=ctx,
               name="KVMN_KT"#,
               #default_bucket_kwargs={'seqlen': params.seqlen}
               )
    #print "net.params.items()=====>"
    #for k, v in net.params.items():
    #    print k, "\t\t", LA.norm(v.asnumpy())
    #print "==========================================================================="
    #for k, v in net.params.items():
    #    print k, "\n", v.asnumpy()
    #    print "                                                                         ---->",\
    #        k, nd.norm(v).asnumpy()
    #net.print_stat()

    ### ================================== Reading data ==================================
    dat = DATA(n_question = params.n_question, seqlen=params.seqlen, separate_char=',')
    train_data_path = params.data_dir + "/" + params.data_name + "_train.csv"
    test_data_path = params.data_dir + "/" + params.data_name + "_test.csv"
    train_q_data, train_qa_data = dat.load_data(train_data_path)
    test_q_data, test_qa_data = dat.load_data(test_data_path)
    print "\n"
    print "train_q_data.shape",train_q_data.shape ###(3633, 200) = (#sample, seqlen)
    print "train_qa_data.shape",train_qa_data.shape ###(3633, 200) = (#sample, seqlen)
    print "test_q_data.shape",test_q_data.shape   ###(1566, 200)
    print "test_qa_data.shape", test_qa_data.shape  ###(1566, 200)
    print "\n"

    file_name = 'key_value_' + 'qembed' + str(params.q_embed_dim) + 'qaembed' + str(params.qa_embed_dim) + \
                'qdim' + str(params.q_state_dim) + 'qadim' + str(params.qa_state_dim) + \
                'msize' + str(params.memory_size) + 'mkdim' + str(params.memory_key_state_dim) + 'mvdim' + str(params.memory_value_state_dim) + \
                'h' + str(params.num_heads) + \
                'std' + str(params.init_std) + 'lr' + str(params.init_lr) + 'mmt' + str(params.momentum) + 'gn' + str(params.maxgradnorm)
    ### ================================== start training ==================================
    all_loss = {}
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_test_loss = {}
    all_test_accuracy = {}
    all_test_auc = {}
    optimizer = mx.optimizer.create(name='SGD', learning_rate=params.lr, momentum=params.momentum,
                                    rescale_grad=1.0 / params.batch_size)
    updater = mx.optimizer.get_updater(optimizer)
    if not params.test:
        for idx in xrange(params.max_iter):
            train_loss, train_accuracy, train_auc = train(net, updater, params, train_q_data, train_qa_data, label='Train')
            test_loss, test_accuracy, test_auc = test(net, params, test_q_data, test_qa_data, label='Test')
            print 'epoch', idx + 1
            print "test_auc\t", test_auc, "\ttrain_auc\t", train_auc
            print "test_accuracy\t", test_accuracy, "\ttrain_accuracy\t", train_accuracy
            print "test_loss\t", test_loss, "\ttrain_loss\t", train_loss
            print "learning_rate", params.lr

            m = len(all_loss) + 1
            all_loss[m] = [m, train_loss, test_loss, train_accuracy, test_accuracy, train_auc, test_auc]
            all_test_auc[m] = test_auc
            all_train_auc[m] = train_auc
            all_test_loss[m] = test_loss
            all_train_loss[m] = train_loss
            all_test_accuracy[m] = test_accuracy
            all_train_accuracy[m] = train_accuracy

            net.save_params(dir_path=os.path.join('model', params.save, file_name, str(idx+1)+"_"+str(test_auc)))
            # Learning rate annealing
            if m > 1 and all_loss[m][2] > all_loss[m - 1][2] * 0.9999:
                params.lr = params.lr / 1.5
            if params.lr < 1e-5: break

        f_save_log = open(os.path.join('result', params.save, file_name),'w')
        f_save_log.write("test_auc:\n"+str(all_test_auc) + "\n\n")
        f_save_log.write("train_auc:\n"+str(all_train_auc) + "\n\n")
        f_save_log.write("test_loss:\n"+str(all_test_loss) + "\n\n")
        f_save_log.write("train_loss:\n"+str(all_train_loss) + "\n\n")
        f_save_log.write("test_accuracy:\n"+str(all_test_accuracy) + "\n\n")
        f_save_log.write("train_accuracy:\n"+str(all_train_accuracy) + "\n\n")
        f_save_log.write(str(all_loss)+"\n")
        f_save_log.close()
        print all_loss

    # run -test "embed100cdim100msize128mdim100k10gamma0.9r1w1std0.1lr0.1mmt0.9gn100"
    # python main.py --gpus 0 --k_smallest 5 --gamma 0.9 --init_std 0.05 --init_lr 0.1 --momentum 0.9 --maxgradnorm 50 --test True --show False --vis True
    else:
        net.load_params(name="LRUA_KT", dir_path=os.path.join('model', params.load, file_name))
        train_loss, train_accuracy, train_auc = test(net, params, train_q_data, train_qa_data, label='Train')
        test_loss, test_accuracy, test_auc = test(net, params, test_q_data, test_qa_data, label='Test')

        print "test_auc\t", test_auc, "\ttrain_auc\t", train_auc
        print "test_accuracy\t", test_accuracy, "\ttrain_accuracy\t", train_accuracy
        print "test_loss\t", test_loss, "\ttrain_loss\t", train_loss
        print "learning_rate", params.lr





