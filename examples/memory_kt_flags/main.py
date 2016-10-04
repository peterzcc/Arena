import logging
import argparse
import numpy as np
from load_data import DATA
from arena import Base
from model import MODEL
from arena.utils import *
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

#class LRUAInitializer(mx.init.Xavier):
class LRUAInitializer(mx.initializer.Normal):
    def _init_default(self, name, arr):
        if name == "init_memory":
            arr[:] = numpy.random.normal(loc=0.1, size=arr.shape)
        #elif name == "init_read_content":
        #    arr[:] = numpy.random.normal(loc=0.1, size=arr.shape)
        elif ("read_head" in name or "write_head" in name) and ("init_W_r_focus" in name or "init_W_u_focus" in name):
            arr[:] = numpy.broadcast_to(numpy.arange(arr.shape[-1], 0, -1), arr.shape)
        elif "init_h" in name or "init_c" in name:
            arr[:] = numpy.random.normal(loc=0.1, size=arr.shape)
        else:
            assert False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test MANN.')
    parser.add_argument('--gpus', type=str, default="0", help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--max_iter', type=int, default=100, help='number of iterations')
    parser.add_argument('--num_reads', type=int, default=1, help='number of read tensors')
    parser.add_argument('--num_writes', type=int, default=1, help='number of write tensors')

    parser.add_argument('--test', type=bool, default=False, help='enable testing')
    parser.add_argument('--show', type=bool, default=True, help='print progress')
    parser.add_argument('--vis', type=bool, default=False, help='visualize weights and results')

    #load command: python main.py --test True --vis True

    dataset = "STATICS" #  assist2009_old / assist2009_updated / assist2015 / KDDal0506 / STATICS

    if dataset == "assist2009_old":
        ### assist2009_old
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--embed_dim', type=int, default=100, help='embedding dimensions')
        parser.add_argument('--control_state_dim', type=int, default=100, help='hidden states of the controller')
        parser.add_argument('--memory_size', type=int, default=20, help='memory size')
        parser.add_argument('--memory_state_dim', type=int, default=100, help='internal state dimension')
        parser.add_argument('--k_smallest', type=int, default=1, help='parmeter of k smallest flags for w_lu')
        parser.add_argument('--gamma', type=float, default=0.9, help='hyperparameter of decay W_u')
        parser.add_argument('--controller', type=str, default="FNN", help='type of controller, i.e. LSTM or FNN')

        parser.add_argument('--init_std', type=float, default=0.05, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.1, help='initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')

        parser.add_argument('--n_question', type=int, default=111, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../data/assist2009_old', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2009_old', help='data set name')
        parser.add_argument('--load', type=str, default='assist2009_old', help='model file to load')
        parser.add_argument('--save', type=str, default='assist2009_old', help='path to save model')
    elif dataset == "assist2009_updated":
        ### assist2009_updated
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--embed_dim', type=int, default=100, help='embedding dimensions')
        parser.add_argument('--control_state_dim', type=int, default=100, help='hidden states of the controller')
        parser.add_argument('--memory_size', type=int, default=20, help='memory size')
        parser.add_argument('--memory_state_dim', type=int, default=100, help='internal state dimension')
        parser.add_argument('--k_smallest', type=int, default=1, help='parmeter of k smallest flags for w_lu')
        parser.add_argument('--gamma', type=float, default=0.9, help='hyperparameter of decay W_u')
        parser.add_argument('--controller', type=str, default="FNN", help='type of controller, i.e. LSTM or FNN')

        parser.add_argument('--init_std', type=float, default=0.05, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.5, help='initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=10.0, help='maximum gradient norm')

        parser.add_argument('--n_question', type=int, default=111, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../data/assist2009_updated', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2009_updated', help='data set name')
        parser.add_argument('--load', type=str, default='assist2009_updated', help='model file to load')
        parser.add_argument('--save', type=str, default='assist2009_updated', help='path to save model')

    elif dataset == "assist2015":
    ### assistment2015
        parser.add_argument('--batch_size', type=int, default=50, help='the batch size')
        parser.add_argument('--embed_dim', type=int, default=100, help='embedding dimensions')
        parser.add_argument('--control_state_dim', type=int, default=100, help='hidden states of the controller')
        parser.add_argument('--memory_size', type=int, default=20, help='memory size')
        parser.add_argument('--memory_state_dim', type=int, default=100, help='internal state dimension')
        parser.add_argument('--k_smallest', type=int, default=1, help='parmeter of k smallest flags for w_lu')
        parser.add_argument('--gamma', type=float, default=0.9, help='hyperparameter of decay W_u')
        parser.add_argument('--controller', type=str, default="FNN", help='type of controller, i.e. LSTM or FNN')

        parser.add_argument('--init_std', type=float, default=0.05, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')

        parser.add_argument('--n_question', type=int, default=100, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../data/assist2015', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2015', help='data set name')
        parser.add_argument('--load', type=str, default='assist2015', help='model file to load')
        parser.add_argument('--save', type=str, default='assist2015', help='path to save model')

    elif dataset == "KDDal0506":
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--embed_dim', type=int, default=100, help='embedding dimensions')
        parser.add_argument('--control_state_dim', type=int, default=100, help='hidden states of the controller')
        parser.add_argument('--memory_size', type=int, default=50, help='memory size')
        parser.add_argument('--memory_state_dim', type=int, default=100, help='internal state dimension')
        parser.add_argument('--k_smallest', type=int, default=1, help='parmeter of k smallest flags')
        parser.add_argument('--gamma', type=float, default=0.9, help='hyperparameter of decay W_u')
        parser.add_argument('--controller', type=str, default="FNN", help='type of controller, i.e. LSTM or FNN')

        parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.1, help='initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')

        parser.add_argument('--n_question', type=int, default=436, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=200, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../data/KDDal0506', help='data directory')
        parser.add_argument('--data_name', type=str, default='KDDal0506', help='data set name')
        parser.add_argument('--load', type=str, default='KDDal0506', help='model file to load')
        parser.add_argument('--save', type=str, default='KDDal0506', help='path to save model')
    elif dataset == "STATICS":
        ### STATICS
        parser.add_argument('--batch_size', type=int, default=10, help='the batch size')
        parser.add_argument('--embed_dim', type=int, default=100, help='embedding dimensions')
        parser.add_argument('--control_state_dim', type=int, default=100, help='hidden states of the controller')
        parser.add_argument('--memory_size', type=int, default=128, help='memory size')
        parser.add_argument('--memory_state_dim', type=int, default=100, help='internal state dimension')
        parser.add_argument('--k_smallest', type=int, default=2, help='parmeter of k smallest flags')
        parser.add_argument('--gamma', type=float, default=0.9, help='hyperparameter of decay W_u')
        parser.add_argument('--controller', type=str, default="FNN", help='type of controller, i.e. LSTM or FNN')

        parser.add_argument('--init_std', type=float, default=0.01, help='weight initialization std')
        parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
        parser.add_argument('--maxgradnorm', type=float, default=10.0, help='maximum gradient norm')

        parser.add_argument('--n_question', type=int, default=1223, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=100, help='the allowed maximum length of a sequence')
        parser.add_argument('--data_dir', type=str, default='../data/STATICS', help='data directory')
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

    ### ================================== controller choice ==================================
    if params.controller == "FNN":
        data_shapes = {'data': (params.seqlen, params.batch_size),
                       'target': (params.seqlen, params.batch_size),
                       'init_memory': (params.memory_size, params.memory_state_dim),
                       #'init_read_content': (params.batch_size, params.num_reads, params.memory_state_dim),
                       'MANN->write_head:init_W_r_focus': (params.num_writes, params.memory_size),
                       'MANN->write_head:init_W_u_focus': (params.num_writes, params.memory_size)}
        learn_init_keys = ['init_memory',
                           #'init_read_content',
                           'MANN->write_head:init_W_r_focus',
                           'MANN->write_head:init_W_u_focus']
    elif params.controller == "LSTM":
        data_shapes = {'data': (params.seqlen, params.batch_size),
                       'target': (params.seqlen, params.batch_size),
                       'init_memory': (params.memory_size, params.memory_state_dim),
                       #'init_read_content': (params.batch_size, params.num_reads, params.memory_state_dim),
                       'MANN->write_head:init_W_r_focus': (params.num_writes, params.memory_size),
                       'MANN->write_head:init_W_u_focus': (params.num_writes, params.memory_size),
                       'controller->layer0:init_h': (params.control_state_dim),
                       'controller->layer0:init_c': (params.control_state_dim)}
        learn_init_keys = ['init_memory',
                           #'init_read_content',
                           'MANN->write_head:init_W_r_focus',
                           'MANN->write_head:init_W_u_focus',
                           'controller->layer0:init_h',
                           'controller->layer0:init_c']

    ### ================================== start training ==================================

    file_name = 'MANN_learn_init_embed' + str(params.embed_dim) + 'cdim' + str(params.control_state_dim) + \
                'msize' + str(params.memory_size) + 'mdim' + str(params.memory_state_dim) + \
                'k' + str(params.k_smallest) + 'gamma' + str(params.gamma) + \
                'r' + str(params.num_reads) + 'w' + str(params.num_writes) + \
                'std' + str(params.init_std) + 'lr' + str(params.init_lr) + \
                'mmt' + str(params.momentum) + 'gn' + str(params.maxgradnorm) + \
                'contr' + params.controller

    dat = DATA(n_question=params.n_question, seqlen=params.seqlen, separate_char=',')
    all_loss = {}
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_test_loss = {}
    all_test_accuracy = {}
    all_test_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}

    if not params.test:
        #traing with five fold cross validation
        five_fold = False
        if five_fold == True:
            for k in range(1, 6):
                g_model = MODEL(n_question=params.n_question,
                                seqlen=params.seqlen,
                                batch_size=params.batch_size,
                                embed_dim=params.embed_dim,
                                control_state_dim=params.control_state_dim,
                                memory_size=params.memory_size,
                                memory_state_dim=params.memory_state_dim,
                                k_smallest=params.k_smallest,
                                gamma=params.gamma,
                                controller=params.controller,
                                num_reads=params.num_reads,
                                num_writes=params.num_writes)
                #initializer = LRUAInitializer(factor_type="in", rnd_type="gaussian", magnitude=2)
                initializer = LRUAInitializer(sigma=params.init_std)
                net = Base(sym_gen=g_model.sym_gen(),
                           data_shapes=data_shapes,
                           learn_init_keys=learn_init_keys,
                           initializer=initializer,
                           #default_bucket_kwargs={'seqlen': params.seqlen},
                           ctx=ctx,
                           name="LRUA_KT")
                # print "net.params.items()=====>"
                # for k, v in net.params.items():
                #    print k, "\t\t", LA.norm(v.asnumpy())
                # print "==========================================================================="
                # for k, v in net.params.items():
                #    print k, "\n", v.asnumpy()
                #    print "                                                                         ---->",\
                #        k, nd.norm(v).asnumpy()
                net.print_stat()

                train_data_path = params.data_dir + "/" + params.data_name + "_train" + str(k) + ".csv"
                valid_data_path = params.data_dir + "/" + params.data_name + "_valid" + str(k) + ".csv"
                test_data_path = params.data_dir + "/" + params.data_name + "_test.csv"
                train_data = dat.load_data(train_data_path)
                valid_data = dat.load_data(valid_data_path)
                test_data = dat.load_data(test_data_path)
                print "\n"
                print "train_data.shape", train_data.shape  ###(3633, 200) = (#sample, seqlen)
                print "valid_data.shape", valid_data.shape
                print "test_data.shape", test_data.shape  ###(1566, 200)
                print "\n"

                optimizer = mx.optimizer.create(name='SGD', learning_rate=params.lr, momentum=params.momentum,
                                                rescale_grad=1.0 / params.batch_size)
                updater = mx.optimizer.get_updater(optimizer)

                for idx in xrange(params.max_iter):
                    train_loss, train_accuracy, train_auc = train(net, updater, params, train_data, label='Train')
                    valid_loss, valid_accuracy, valid_auc = test(net, updater, params, valid_data, label='Valid')
                    test_loss, test_accuracy, test_auc = test(net, params, test_data, label='Test')
                    print 'epoch', idx + 1
                    print "valid_auc", valid_auc, "\ttest_auc\t", test_auc, "\ttrain_auc\t", train_auc
                    print "valid_accuracy", valid_accuracy, "\ttest_accuracy\t", test_accuracy, "\ttrain_accuracy\t", train_accuracy
                    print "valid_loss", valid_loss, "\ttest_loss\t", test_loss, "\ttrain_loss\t", train_loss
                    print "learning_rate", optimizer.lr


                    m = len(all_loss) + 1
                    all_loss[m] = [m, train_loss, valid_loss, test_loss, train_accuracy, valid_accuracy, test_accuracy, train_auc, valid_auc, test_auc]
                    all_valid_auc[m] = valid_auc
                    all_test_auc[m] = test_auc
                    all_train_auc[m] = train_auc
                    all_valid_loss[m] = valid_loss
                    all_test_loss[m] = test_loss
                    all_train_loss[m] = train_loss
                    all_valid_accuracy[m] = valid_accuracy
                    all_test_accuracy[m] = test_accuracy
                    all_train_accuracy[m] = train_accuracy

                    net.save_params(dir_path=os.path.join('model', params.save, file_name, str(idx + 1)+"_"+str(test_auc)))
                    # Learning rate annealing
                    if 0 == (idx + 1) % 10:
                        optimizer.lr = optimizer.lr / 1.5
                    #if m > 1 and all_loss[m][2] > all_loss[m - 1][2] * 0.9999:
                    #    params.lr = params.lr / 1.5
                    #if params.lr < 1e-5: break
                ### save and print all results
                f_save_log = open(os.path.join('result', params.save, file_name),'a')
                f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
                f_save_log.write("test_auc:\n"+str(all_test_auc) + "\n\n")
                f_save_log.write("train_auc:\n"+str(all_train_auc) + "\n\n")
                f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
                f_save_log.write("test_loss:\n"+str(all_test_loss) + "\n\n")
                f_save_log.write("train_loss:\n"+str(all_train_loss) + "\n\n")
                f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
                f_save_log.write("test_accuracy:\n"+str(all_test_accuracy) + "\n\n")
                f_save_log.write("train_accuracy:\n"+str(all_train_accuracy) + "\n\n")
                f_save_log.write(str(all_loss)+"\n")
                f_save_log.close()
                print all_loss
        else:
            g_model = MODEL(n_question=params.n_question,
                            seqlen=params.seqlen,
                            batch_size=params.batch_size,
                            embed_dim=params.embed_dim,
                            control_state_dim=params.control_state_dim,
                            memory_size=params.memory_size,
                            memory_state_dim=params.memory_state_dim,
                            k_smallest=params.k_smallest,
                            gamma=params.gamma,
                            controller=params.controller,
                            num_reads=params.num_reads,
                            num_writes=params.num_writes)
            # initializer = LRUAInitializer(factor_type="in", rnd_type="gaussian", magnitude=2)
            initializer = LRUAInitializer(sigma=params.init_std)
            net = Base(sym_gen=g_model.sym_gen(),
                       data_shapes=data_shapes,
                       learn_init_keys=learn_init_keys,
                       initializer=initializer,
                       # default_bucket_kwargs={'seqlen': params.seqlen},
                       ctx=ctx,
                       name="LRUA_KT")
            # print "net.params.items()=====>"
            # for k, v in net.params.items():
            #    print k, "\t\t", LA.norm(v.asnumpy())
            # print "==========================================================================="
            # for k, v in net.params.items():
            #    print k, "\n", v.asnumpy()
            #    print "                                                                         ---->",\
            #        k, nd.norm(v).asnumpy()
            net.print_stat()

            train_data_path = params.data_dir + "/" + params.data_name + "_train.csv"
            test_data_path = params.data_dir + "/" + params.data_name + "_test.csv"
            train_data = dat.load_data(train_data_path)
            test_data = dat.load_data(test_data_path)
            print "\n"
            print "train_data.shape", train_data.shape  ###(3633, 200) = (#sample, seqlen)
            print "test_data.shape", test_data.shape  ###(1566, 200)
            print "\n"
            optimizer = mx.optimizer.create(name='SGD', learning_rate=params.lr, momentum=params.momentum,
                                            rescale_grad=1.0 / params.batch_size)
            updater = mx.optimizer.get_updater(optimizer)

            for idx in xrange(params.max_iter):
                train_loss, train_accuracy, train_auc = train(net, updater, params, train_data, label='Train')
                test_loss, test_accuracy, test_auc = test(net, params, test_data, label='Test')
                print 'epoch', idx + 1
                print "test_auc", test_auc, "\ttrain_auc\t", train_auc
                print "test_accuracy", test_accuracy, "\ttrain_accuracy\t", train_accuracy
                print "test_loss", test_loss, "\ttrain_loss\t", train_loss
                print "learning_rate", optimizer.lr

                m = len(all_loss) + 1
                all_loss[m] = [m, train_loss, test_loss, train_accuracy, test_accuracy,
                               train_auc, test_auc]
                all_test_auc[m] = test_auc
                all_train_auc[m] = train_auc
                all_test_loss[m] = test_loss
                all_train_loss[m] = train_loss
                all_test_accuracy[m] = test_accuracy
                all_train_accuracy[m] = train_accuracy

                net.save_params(dir_path=os.path.join('model', params.save, file_name, str(idx + 1) + "_" + str(test_auc)))
                # Learning rate annealing
                if 0 == (idx + 1) % 10:
                    optimizer.lr = optimizer.lr / 1.5
                #if m > 1 and all_loss[m][2] > all_loss[m - 1][2] * 0.9999:
                #    params.lr = params.lr / 1.5
                #if params.lr < 1e-5: break
            ### save and print all results
            f_save_log = open(os.path.join('result', params.save, file_name), 'a')
            f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
            f_save_log.write("test_auc:\n" + str(all_test_auc) + "\n\n")
            f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
            f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
            f_save_log.write("test_loss:\n" + str(all_test_loss) + "\n\n")
            f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
            f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
            f_save_log.write("test_accuracy:\n" + str(all_test_accuracy) + "\n\n")
            f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
            f_save_log.write(str(all_loss) + "\n")
            f_save_log.close()
            print all_loss

    # run -test "embed100cdim100msize128mdim100k10gamma0.9r1w1std0.1lr0.1mmt0.9gn100"
    # python main.py --gpus 0 --k_smallest 5 --gamma 0.9 --init_std 0.05 --init_lr 0.1 --momentum 0.9 --maxgradnorm 50 --test True --show False --vis True
    else:
        g_model = MODEL(n_question=params.n_question,
                        seqlen=params.seqlen,
                        batch_size=params.batch_size,
                        embed_dim=params.embed_dim,
                        control_state_dim=params.control_state_dim,
                        memory_size=params.memory_size,
                        memory_state_dim=params.memory_state_dim,
                        k_smallest=params.k_smallest,
                        gamma=params.gamma,
                        controller=params.controller,
                        num_reads=params.num_reads,
                        num_writes=params.num_writes)
        # initializer = LRUAInitializer(factor_type="in", rnd_type="gaussian", magnitude=2)
        initializer = LRUAInitializer(sigma=params.init_std)
        net = Base(sym_gen=g_model.sym_gen(),
                   data_shapes=data_shapes,
                   learn_init_keys=learn_init_keys,
                   initializer=initializer,
                   # default_bucket_kwargs={'seqlen': params.seqlen},
                   ctx=ctx,
                   name="LRUA_KT")
        ### change the loading model file
        epoch_file_name = os.path.join(file_name,"33_0.868647189405")
        load_path = os.path.join('model', params.load, epoch_file_name)
        net.load_params(name="LRUA_KT", dir_path=load_path)
        params.load_path = load_path
        train_data_path = params.data_dir + "/" + params.data_name + "_train.csv"
        test_data_path = params.data_dir + "/" + params.data_name + "_test.csv"
        train_data = dat.load_data(train_data_path)
        test_data = dat.load_data(test_data_path)

        train_loss, train_accuracy, train_auc = test(net, params, train_data, label='Train')
        test_loss, test_accuracy, test_auc = test(net, params, test_data, label='Test')
        print "test_auc\t", test_auc, "\ttrain_auc\t", train_auc
        print "test_accuracy\t", test_accuracy, "\ttrain_accuracy\t", train_accuracy
        print "test_loss\t", test_loss, "\ttrain_loss\t", train_loss






