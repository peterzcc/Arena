from arena.helpers.visualization import *
import numpy as np


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

def vis_matrix(params, pred_3d, target, batch_id):
    # :Parameter pred_3d : Shape (seqlen, batch_size , n_question)
    # :Parameter target :  Shape (seqlen, batch_size) = (200*32)
    vis_pred_all = pred_3d[:,batch_id,:] # Shape ( seqlen, n_question ) when vis using .T
    target = target[:,batch_id]

    vis_target_one_hot = np.zeros((params.seqlen, params.n_question))
    vis_pred_one_hot = np.zeros((params.seqlen, params.n_question))
    target = target.astype(np.int)

    zero_index = np.flatnonzero(target == 0)
    non_zero_index = np.flatnonzero(target)
    next_label = (target - 1) % params.n_question  # Shape (batch_size*seqlen*N, )
    truth = (target - 1) / params.n_question  # Shape (batch_size*seqlen*N, )
    next_label[zero_index] = 0
    truth_wrong = np.flatnonzero(truth == 0)
    truth[truth_wrong] = -1
    truth[zero_index] = 0

    next_label = next_label.tolist()
    vis_target_one_hot[np.arange(len(next_label)), next_label] = truth[np.arange(len(next_label))]
    vis_pred_one_hot[np.arange(len(next_label)), next_label] = vis_pred_all[np.arange(len(next_label)), next_label]
    vis_pred_one_hot[zero_index,0] = 0.0

    return vis_pred_all, vis_pred_one_hot, vis_target_one_hot

def vis_pred_target_1d(params, pred_3d, target, batch_id):
    pred_one_seq = pred_3d[:, batch_id, :]  # Shape ( seqlen, n_question ) when vis using .T
    target_one_seq = target[:, batch_id]
    target_one_seq = target_one_seq.astype(np.int)

    ### tailor sequence to its actual length
    zero_index = np.flatnonzero(target_one_seq == 0)
    non_zero_index = np.flatnonzero(target_one_seq)
    print "non_zero_index", non_zero_index
    target_one_seq_tailored = target_one_seq[non_zero_index]
    print "target_one_seq_tailored", target_one_seq_tailored
    seq_len = len(target_one_seq_tailored)
    print "seq_len", seq_len

    next_label = (target_one_seq_tailored - 1) % params.n_question
    truth = (target_one_seq_tailored - 1) / params.n_question

    vis_pred_1d = pred_one_seq[np.arange(seq_len), next_label]
    vis_target_1d = truth
    print "vis_pred_1d",vis_pred_1d
    print "vis_target_1d", vis_target_1d
    pred_target_tailored_1d = np.stack((vis_pred_1d, vis_target_1d))

    return pred_target_tailored_1d


def vis_kt_one(params, idx, pred, target, control_state, read_content, read_focus, write_focus):
    # read_focus -- Shape ( sequence length, batch size, memory size )
    # write_focus -- Shape ( sequence length, batch size, memory size )
    # vis_pred_all -- Shape (sequence length, batch size, n_q)
    # vis_pred_one_hot -- Shape (sequence length, batch size, n_q)
    # vis_target_one_hot -- Shape (sequence length, batch size, n_q)
    # control_state -- Shape (sequence length, batch size, control state dim)
    # read_content -- Shape (sequence length, batch size, memory state dim)
    """
    print "\nread_focus.shape", read_focus.shape # Shape
    print read_focus[:, 0, :].max(axis=1), read_focus[:, 0, :].max(axis=1).shape
    print "\nwrite_focus.shape", write_focus.shape
    print write_focus[:, 0, :].max(axis=1), write_focus[:, 0, :].max(axis=1).shape
    """

    for batch_iter in range(params.batch_size):
        """
        pred_target_tailored_1d = vis_pred_target_1d(params, pred_3d, target, batch_iter)
        fig2 = plt.figure(batch_iter + idx * params.batch_size + 100000, figsize=(200, 150))
        plt.matshow(pred_target_tailored_1d)
        plt.colorbar()
        plt.show
        """

        vis_pred = pred.reshape((params.seqlen, params.batch_size))
        fig = plt.figure(batch_iter + idx * params.batch_size, figsize=(200, 150))
        xlabel_name = "sequence length"
        # PLT2Vis.display(data=vis_pred_all.T, x_label="sequence length", y_label="n_q", win_name="prediction_all", scale=0.5)
        a = fig.add_subplot(2, 4, 1)
        plt.imshow(vis_pred[:,0].T)
        a.set_title('vis_pred_all')
        plt.xlabel(xlabel_name)
        plt.ylabel("")
        plt.colorbar(orientation='horizontal')
        # PLT2Vis.display(data=vis_pred_one_hot.T, x_label="sequence length", y_label="n_q", win_name="prediction_one_hot", scale=0.5)
        a = fig.add_subplot(2, 4, 2)
        plt.imshow(target[:,0].T)
        a.set_title('vis_pred_one_hot')
        plt.xlabel(xlabel_name)
        plt.ylabel("")
        plt.colorbar(orientation='horizontal')

        # PLT2Vis.display(data=control_state[:, 0, :].T, x_label="sequence length", y_label="control state dim", win_name="control_state", scale=0.5)
        a = fig.add_subplot(2, 4, 5)
        plt.imshow(control_state[:, 0, :].T)
        a.set_title('control_state')
        plt.xlabel(xlabel_name)
        plt.ylabel("control state dim")
        plt.colorbar(orientation='horizontal')
        # PLT2Vis.display(data=read_content[:, 0, :].T, x_label="sequence length", y_label="memory state dim", win_name="read_content", scale=0.5)
        a = fig.add_subplot(2, 4, 6)
        plt.imshow(read_content[:, 0, :].T)
        a.set_title('read_content')
        plt.xlabel(xlabel_name)
        plt.ylabel("memory state dim")
        plt.colorbar(orientation='horizontal')
        # PLT2Vis.display(data=read_focus[:, 0, :].T, x_label="sequence length", y_label="memory size", win_name="read_focus", scale=0.5)
        a = fig.add_subplot(2, 4, 7)
        plt.imshow(read_focus[:, 0, :].T)
        a.set_title('key_read_focus')
        plt.xlabel(xlabel_name)
        plt.ylabel("memory size")
        plt.colorbar(orientation='horizontal')
        # PLT2Vis.display(data=write_focus[:, 0, :].T, x_label="sequence length", y_label="memory size", win_name="write_focus", scale=0.5)
        a = fig.add_subplot(2, 4, 8)
        plt.imshow(write_focus[:, 0, :].T)
        a.set_title('value_read_focus')
        plt.xlabel(xlabel_name)
        plt.ylabel("memory size")
        plt.colorbar(orientation='horizontal')
        plt.show()
        # plt.savefig(os.path.join(params.load_path, str(batch_iter+idx*params.batch_size)))




def vis_weight(params, idx, target, read_focus, write_focus):
    for batch_iter in range(params.batch_size):
        target_one_seq = target[:, batch_iter]
        print "target_one_seq", target_one_seq
        non_zero_index = np.flatnonzero(target_one_seq != -1.0)
        print "non_zero_index", non_zero_index
        target_one_seq_tailored = target_one_seq[non_zero_index]
        seq_len = len(target_one_seq_tailored)

        fig = plt.figure(batch_iter + idx * params.batch_size, figsize=(200, 150))
        xlabel_name = "sequence length"


        a = fig.add_subplot(1, 1, 1)
        ma = a.matshow(read_focus[:seq_len, batch_iter, :].T)
        a.set_title('key_read_focus')
        plt.xlabel(xlabel_name)
        plt.ylabel("memory size")
        fig.colorbar(ma)

        # PLT2Vis.display(data=write_focus[:, 0, :].T, x_label="sequence length", y_label="memory size", win_name="write_focus", scale=0.5)
        """
        b = fig.add_subplot(2, 2, 4)
        mb = b.matshow(write_focus[:seq_len, batch_iter, :].T)
        b.set_title('value_read_focus')
        plt.xlabel(xlabel_name)
        plt.ylabel("memory size")
        fig.colorbar(mb)
        """
        plt.show()