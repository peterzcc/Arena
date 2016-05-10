import mxnet as mx
import mxnet.ndarray as nd
import numpy
import cv2
import json
import time
import argparse
import sys
from core import *
from arena.iterators import TrackingIterator
from arena.helpers.visualization import *
from arena.helpers.tracking import *
from scipy.stats import entropy


root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)
root.setLevel(logging.DEBUG)
mx.random.seed(100)


#TODO sym_out can be replaced by mx.symbol.list_outputs() in the future

def parse_tracker_outputs(outputs, sym_out, total_timesteps, attention_steps, memory_size,
                          cf_handler, scoremap_processor,
                          glimpse_data_shape=None, parse_all=False):
    ret = OrderedDict()
    ret['pred_rois'] = numpy.zeros((total_timesteps, 4), dtype=numpy.float32)
    ret['search_rois'] = numpy.zeros((total_timesteps, attention_steps, 4), dtype=numpy.float32)
    ret['trans_search_rois'] = numpy.zeros((total_timesteps, attention_steps - 1, 4),
                                           dtype=numpy.float32)
    ret['trans_pred_rois'] = numpy.zeros((total_timesteps, 4),
                                           dtype=numpy.float32)
    ret['bb_regress_loss'] = numpy.zeros((total_timesteps, 4), dtype=numpy.float32)

    ret['read_controls'] = numpy.empty((total_timesteps, attention_steps), dtype=numpy.float32)
    ret['write_controls'] = numpy.empty((total_timesteps,), dtype=numpy.float32)
    ret['read_controls_prob'] = numpy.empty((total_timesteps, attention_steps, memory_size),
                                              dtype=numpy.float32)
    ret['write_controls_prob'] = numpy.empty((total_timesteps, 3), dtype=numpy.float32)

    if glimpse_data_shape is not None:
        ret['pred_glimpse_data'] = numpy.zeros((total_timesteps, ) + glimpse_data_shape,
                                        dtype=numpy.float32)
    counter_checking = {'pred_rois': 0,
                        'trans_search_rois': 0,
                        'search_rois': 0,
                        'trans_pred_rois': 0,
                        'bb_regress_loss': 0,
                        'read_controls': 0,
                        'write_controls': 0,
                        'read_controls_prob': 0,
                        'write_controls_prob': 0}

    counter_ground_truth = {'pred_rois': total_timesteps * 2,
                            'trans_search_rois': total_timesteps * (attention_steps - 1) * 2,
                            'search_rois': total_timesteps * attention_steps * 2,
                            'trans_pred_rois': total_timesteps * 2,
                            'bb_regress_loss': total_timesteps,
                            'read_controls': total_timesteps * attention_steps,
                            'write_controls': total_timesteps,
                            'read_controls_prob': total_timesteps * attention_steps,
                            'write_controls_prob': total_timesteps}

    if parse_all:
        ret['attention_scoremap'] = numpy.empty((total_timesteps, attention_steps) +
                                                cf_handler.scoremap_shape, dtype=numpy.float32)
        ret['processed_scoremap'] = numpy.empty((total_timesteps, attention_steps) +
                                                (scoremap_processor.dim_out[0] * scoremap_processor.scale_num,
                                                 scoremap_processor.dim_out[1],
                                                 scoremap_processor.dim_out[2]), dtype=numpy.float32)
        counter_checking['attention_scoremap'] = 0
        counter_checking['processed_scoremap'] = 0
        counter_ground_truth['attention_scoremap'] = total_timesteps * attention_steps
        counter_ground_truth['processed_scoremap'] = total_timesteps * attention_steps

    for i, (key, output) in enumerate(zip(sym_out.keys(), outputs)):
        if 'glimpse_pred_t' in key:
            timestamp = get_timestamp(key)
            if 'center' in key:
                ret['pred_rois'][timestamp, 0:2] = output.asnumpy()
                counter_checking['pred_rois'] += 1
            elif 'size' in key:
                ret['pred_rois'][timestamp, 2:4] = output.asnumpy()
                counter_checking['pred_rois'] += 1
            elif 'data' in key:
                if glimpse_data_shape is not None:
                    ret['pred_glimpse_data'][timestamp, :, :, :, :] = output.asnumpy()
        elif 'real_search' in key:
            timestamp = get_timestamp(key)
            attention_step = get_attention_step(key)
            if 'center' in key:
                ret['search_rois'][timestamp, attention_step, 0:2] = output.asnumpy()
                counter_checking['search_rois'] += 1
            elif 'size' in key:
                ret['search_rois'][timestamp, attention_step, 2:4] = output.asnumpy()
                counter_checking['search_rois'] += 1
        elif 'read:chosen_ind' in key:
            timestamp = get_timestamp(key)
            attention_step = get_attention_step(key)
            if 'action' in key:
                ret['read_controls'][timestamp, attention_step] = output.asnumpy()
                counter_checking['read_controls'] += 1
            elif 'prob' in key:
                ret['read_controls_prob'][timestamp, attention_step] = output.asnumpy()
                counter_checking['read_controls_prob'] += 1
        elif 'write:control_flag' in key:
            timestamp = get_timestamp(key)
            if 'action' in key:
                ret['write_controls'][timestamp] = output.asnumpy()
                counter_checking['write_controls'] += 1
            elif 'prob' in key:
                ret['write_controls_prob'][timestamp] = output.asnumpy()
                counter_checking['write_controls_prob'] += 1
        elif 'bb_regress_loss' in key:
            timestamp = get_timestamp(key)
            ret['bb_regress_loss'][timestamp, :] = output.asnumpy()
            counter_checking['bb_regress_loss'] += 1
        elif 'trans_search' in key:
            timestamp = get_timestamp(key)
            attention_step = get_attention_step(key)
            if 'center' in key:
                ret['trans_search_rois'][timestamp, attention_step, 0:2] = output.asnumpy()
                counter_checking['trans_search_rois'] += 1
            elif 'size' in key:
                ret['trans_search_rois'][timestamp, attention_step, 2:4] = output.asnumpy()
                counter_checking['trans_search_rois'] += 1
        elif 'trans_pred' in key:
            if 'center' in key:
                ret['trans_pred_rois'][timestamp, 0:2] = output.asnumpy()[:]
                counter_checking['trans_pred_rois'] += 1
            elif 'size' in key:
                ret['trans_pred_rois'][timestamp, 2:4] = output.asnumpy()[:]
                counter_checking['trans_pred_rois'] += 1
        if parse_all:
            if 'attention_scoremap' in key:
                timestamp = get_timestamp(key)
                attention_step = get_attention_step(key)
                ret['attention_scoremap'][timestamp, attention_step, :, :, :, :] = output.asnumpy()
                counter_checking['attention_scoremap'] += 1
            elif 'processed_scoremap' in key:
                timestamp = get_timestamp(key)
                attention_step = get_attention_step(key)
                ret['processed_scoremap'][timestamp, attention_step, :, :, :] = output.asnumpy()[0]
                counter_checking['processed_scoremap'] += 1

    assert counter_checking == counter_ground_truth, "Find %s but expected %s" \
                                                     % (counter_checking, counter_ground_truth)
    return ret


def compute_tracking_score(pred_rois, truth_rois, thresholds=(0.5, 0.7, 0.8, 0.9),
                           failure_penalty=-3.0, level_reward=1.0):
    assert pred_rois.shape == truth_rois.shape
    assert len(thresholds) > 1
    overlapping_ratios = cal_rect_int(pred_rois, truth_rois)
    thresholds = sorted(thresholds)
    scores = (failure_penalty * (overlapping_ratios < thresholds[0])).astype(numpy.float32)
    for i, threshold in enumerate(thresholds):
        scores += (level_reward * (overlapping_ratios >= threshold)).astype(numpy.float32)
    return scores + overlapping_ratios


def get_backward_input(init_shapes, scores, baselines, total_timesteps, attention_steps):
    assert scores.shape == baselines.shape
    assert 1 == len(scores.shape)
    assert scores.shape[0] == total_timesteps
    backward_inputs = OrderedDict()
    scores = numpy.cumsum(scores[::-1], axis=0)[::-1]
    advantages = (scores - baselines)
    counter_checking = {'trans_search': 0,
                        'read:chosen_ind': 0,
                        'write:control_flag': 0}
    counter_ground_truth = {'trans_search': total_timesteps * (attention_steps - 1) * 2,
                            'read:chosen_ind': total_timesteps * attention_steps,
                            'write:control_flag': total_timesteps}
    for key in init_shapes.keys():
        if 'score' in key:
            timestamp = get_timestamp(key)
            if 'trans_search' in key:
                backward_inputs[key] = advantages[timestamp]/1000
                counter_checking['trans_search'] += 1
            elif 'trans_pred' in key:
                assert False
            elif 'read:chosen_ind' in key:
                backward_inputs[key] = advantages[timestamp]/100
                counter_checking['read:chosen_ind'] += 1
            elif 'write:control_flag' in key:
                if timestamp < total_timesteps - 1:
                    backward_inputs[key] = advantages[timestamp + 1]
                else:
                    backward_inputs[key] = 0
                counter_checking['write:control_flag'] += 1
            else:
                raise NotImplementedError, 'Only support %s, find key="%s"' \
                                           % (str(counter_checking.keys()), key)
    assert counter_checking == counter_ground_truth, "Find %s but expected %s" \
                                                     %(counter_checking, counter_ground_truth)
    return backward_inputs


parser = argparse.ArgumentParser(description='Script to train the tracking agent.')
parser.add_argument('-d', '--dir-path', required=False, type=str, default='tracking-model-new',
                    help='Saving directory of model files.')
parser.add_argument('-s', '--sequence-path', required=False, type=str,
                    default='D:\\HKUST\\2-2\\learning-to-track\\datasets\\training_for_otb100\\training_otb.lst',
                    help='Saving directory of model files.')
parser.add_argument('-v', '--visualization', required=False, type=int, default=0,
                    help='Visualize the runs.')
parser.add_argument('--train-epoch-num', required=False, type=int, default=200,
                    help='Total training epochs')
parser.add_argument('--train-iter-num', required=False, type=int, default=5000,
                    help='Total iterations for each epoch')
parser.add_argument('--lr', required=False, type=float, default=1E-4,
                    help='Learning rate of the RMSPropNoncentered optimizer')
parser.add_argument('--eps', required=False, type=float, default=1E-6,
                    help='Eps of the RMSPropNoncentered optimizer')
parser.add_argument('--clip-gradient', required=False, type=float, default=None,
                    help='Clip threshold of the RMSPropNoncentered optimizer')
parser.add_argument('--gamma1', required=False, type=float, default=0.95,
                    help='Use Double DQN')
parser.add_argument('--wd', required=False, type=float, default=0.0,
                    help='Weight of the L2 Regularizer')
parser.add_argument('-c', '--ctx', required=False, type=str, default='gpu',
                    help='Running Context. E.g `-c gpu` or `-c gpu1` or `-c cpu`')
parser.add_argument('--roll-out', required=False, type=int, default=3,
                    help='Eps of the epsilon-greedy policy at the beginning')
parser.add_argument('--scale-num', required=False, type=int, default=2,
                    help='Scale number of the glimpse sector')
parser.add_argument('--scale-mult', required=False, type=float, default=1.5,
                    help='Scale multiple of the glimpse sector')
parser.add_argument('--init-scale', required=False, type=float, default=1.7,
                    help='Initial scale of the glimpse sector')
parser.add_argument('--cf-sigma-factor', required=False, type=float, default=20,
                    help='Gaussian sigma factor of the correlation filter')
parser.add_argument('--cf-regularizer', required=False, type=float, default=0.01,
                    help='Regularizer of the correlation filter')
parser.add_argument('--default-update-factor', required=False, type=float, default=0.1,
                    help='Default update factor of the correlation filter')
parser.add_argument('--scoremap-num', required=False, type=int, default=4,
                    help='Number of filters for scoremap')
parser.add_argument('--memory-size', required=False, type=int, default=3,
                    help='Size of the memory unit')
parser.add_argument('--attention-steps', required=False, type=int, default=3,
                    help='Steps of recurrent attention')
parser.add_argument('--sample-length', required=False, type=int, default=16,
                    help='Length of the sampling sequence')
parser.add_argument('--BPTT-length', required=False, type=int, default=15,
                    help='Length of each BPTT step')
parser.add_argument('--interval-step', required=False, type=int, default=1,
                    help='Interval of the sampling sequence')
parser.add_argument('--baseline-lr', required=False, type=float, default=0.001,
                    help='Steps of recurrent attention')
parser.add_argument('--optimizer', required=False, type=str, default="RMSPropNoncentered",
                    help='type of optimizer')
parser.add_argument('--mode', required=False, type=str, choices=['train', 'test'], default='train')
args = parser.parse_args()

ctx = parse_ctx(args.ctx)
ctx = mx.Context(*ctx[0])
logging.info("Arguments:")
for k, v in vars(args).items():
    logging.info("   %s = %s" %(k, v))
quick_save_json(args.dir_path, "args.json", content=vars(args))

sample_length = args.sample_length
BPTT_length = args.BPTT_length

assert (sample_length - 1) % BPTT_length == 0, 'Currently BPTT_length must divide sample_length-1. ' \
                                               'The received sample_length=%d, BPTT_length=%d' %(sample_length,
                                                                                                 BPTT_length)

roll_out_num = args.roll_out
total_epoch_num = args.train_epoch_num
epoch_iter_num = args.train_iter_num
baseline_lr = args.baseline_lr

# Score Related Parameters
thresholds = (0.5, 0.8)
failure_penalty = -0.2
level_reward = 1

# Glimpse Hanlder Parameters
scale_num = args.scale_num
scale_mult = args.scale_mult
init_scale = args.init_scale

# Correlation Filter Handler Parameters
cf_gaussian_sigma_factor = args.cf_sigma_factor
cf_regularizer = args.cf_regularizer

if args.visualization:
    verbose_sym_out = True# Whether to parse all the outputs
else:
    verbose_sym_out = False

if 'test' == args.mode:
    deterministic = True
    random_perturbation = False
else:
    deterministic = False
    random_perturbation = True

scoremap_num_filter = args.scoremap_num

memory_size = args.memory_size
attention_steps = args.attention_steps
image_size = (480, 540)
memory_lstm_props = [LSTMLayerProp(num_hidden=128, dropout=0.),
                     LSTMLayerProp(num_hidden=128, dropout=0.)]
attention_lstm_props = [LSTMLayerProp(num_hidden=128, dropout=0.),
                        LSTMLayerProp(num_hidden=128, dropout=0.)]

sequence_list_path = args.sequence_path
save_dir = args.dir_path

tracking_iterator = TrackingIterator(
    sequence_list_path,
    output_size=image_size,
    resize=True)
glimpse_handler = GlimpseHandler(scale_mult=scale_mult,
                                 scale_num=scale_num,
                                 output_shape=(133, 133),
                                 init_scale=init_scale)
perception_handler = PerceptionHandler(net_type='VGG-M')
cf_handler = CorrelationFilterHandler(rows=64, cols=64,
                                      gaussian_sigma_factor=cf_gaussian_sigma_factor,
                                      regularizer=cf_regularizer,
                                      perception_handler=perception_handler,
                                      glimpse_handler=glimpse_handler)
scoremap_processor = ScoreMapProcessor(dim_in=(96, 64, 64),
                                       num_filter=scoremap_num_filter,
                                       scale_num=scale_num)
memory_handler = MemoryHandler(cf_handler=cf_handler,
                               scoremap_processor=scoremap_processor,
                               memory_size=memory_size,
                               lstm_layer_props=memory_lstm_props)
attention_handler = AttentionHandler(glimpse_handler=glimpse_handler, cf_handler=cf_handler,
                                     memory_handler=memory_handler,
                                     scoremap_processor=scoremap_processor,
                                     total_steps=attention_steps,
                                     lstm_layer_props=attention_lstm_props,
                                     fixed_center_variance=True,
                                     fixed_size_variance=True,
                                     verbose_sym_out=verbose_sym_out)

# 1. Build the memory generator that initialze the memory by analyzing the first frame

memory_generator, mem_sym_out, mem_init_shapes, mem_constant_inputs = \
    build_memory_generator(image_size=image_size,
                           memory_handler=memory_handler,
                           glimpse_handler=glimpse_handler,
                           cf_handler=cf_handler,
                           perception_handler=perception_handler,
                           ctx=ctx)
memory_generator.print_stat()

# 2. Build the tracker following the Perceive, Attend and Memorize procedure
tracker, tracker_sym_out, tracker_init_shapes, tracker_constant_inputs = \
    build_tracker(image_size=image_size,
                  tracking_length=BPTT_length,
                  deterministic=deterministic,
                  memory_handler=memory_handler,
                  glimpse_handler=glimpse_handler,
                  cf_handler=cf_handler,
                  attention_handler=attention_handler,
                  perception_handler=perception_handler,
                  default_update_factor=args.default_update_factor,
                  ctx=ctx)
tracker.print_stat()

tracker.load_params(tracker.name, dir_path="../../../learning-to-track/training-otb-lr0.0001-gamma0.9-mult1.5-init1.7-up-mem4-attend1-score4-len51-blen50", epoch=0)
#tracker.load_params(tracker.name, dir_path="../../tracking-model", epoch=0)

baselines = numpy.zeros((BPTT_length,), dtype=numpy.float32)
optimizer = mx.optimizer.create(name=args.optimizer,
                                learning_rate=args.lr,
                                gamma1=args.gamma1,
                                eps=args.eps,
                                clip_gradient=None,
                                rescale_grad=1.0, wd=args.wd)
updater = mx.optimizer.get_updater(optimizer)
#
# accumulative_grad = OrderedDict()
# for k, v in tracker.params_grad.items():
#     accumulative_grad[k] = nd.empty(shape=v.shape, ctx=v.context)

for epoch in range(total_epoch_num):
    for iter in range(epoch_iter_num):
        seq_images, seq_rois = tracking_iterator.sample(length=sample_length,
                                                        interval_step=args.interval_step,
                                                        verbose=False,
                                                        random_perturbation=random_perturbation)
        # print seq_images.shape
        # print seq_rois.shape
        init_image_ndarray = seq_images[:1].reshape((1,) + seq_images.shape[1:])
        init_roi_ndarray = seq_rois[:1]
        # print init_roi_ndarray.shape
        # print init_image_ndarray.shape
        additional_inputs = OrderedDict()
        additional_inputs['init_image'] = init_image_ndarray
        additional_inputs['init_roi'] = init_roi_ndarray
        # for k, v in mem_constant_inputs.items():
        #    print k, v.shape
        if 0 == iter:
            mem_outputs = memory_generator.forward(is_train=False,
                                                   **(OrderedDict(additional_inputs.items() +
                                                                  mem_constant_inputs.items())))
        else:
            mem_outputs = memory_generator.forward(is_train=False, **additional_inputs)
        init_memory_keys = mem_sym_out.keys()
        init_memory_outputs = mem_outputs
        for bptt_step in range((sample_length-1)/BPTT_length):
            start_indx = BPTT_length*bptt_step + 1
            end_indx = BPTT_length*(bptt_step + 1)
            data_images_ndarray = seq_images[start_indx:(end_indx + 1)].reshape(
                (1, BPTT_length,) + seq_images.shape[1:])
            data_rois_ndarray = seq_rois[start_indx:(end_indx  + 1)].reshape((1, BPTT_length, 4))

            additional_inputs = OrderedDict()
            additional_inputs['data_images'] = data_images_ndarray
            additional_inputs['data_rois'] = data_rois_ndarray
            additional_inputs['init_search_roi'] = init_roi_ndarray
            for i, k in enumerate(init_memory_keys):
                if 'init_memory' in k:
                    additional_inputs[k] = init_memory_outputs[i]

            # avg_scores = numpy.zeros((BPTT_length,), dtype=numpy.float32)
            parsed_outputs_list = []
        #for episode in range(roll_out_num):
            if iter == 0:
                tracker_outputs = tracker.forward(is_train=True, **(OrderedDict(additional_inputs.items() +
                                                                                tracker_constant_inputs.items())))
            else:
                tracker_outputs = tracker.forward(is_train=True, **additional_inputs)
        # else:
        #     tracker_outputs = tracker.forward(is_train=True)
            init_memory_outputs = []
            init_memory_keys = []
            for (k, v) in zip(tracker_sym_out.keys(), tracker_outputs):
                if 'last_step_memory' in k:
                    init_memory_outputs.append(v)
                    init_memory_keys.append(k.replace("last_step_memory", "init_memory"))

            parsed_outputs = parse_tracker_outputs(outputs=tracker_outputs,
                                                   sym_out=tracker_sym_out,
                                                   total_timesteps=BPTT_length,
                                                   attention_steps=attention_steps,
                                                   memory_size=memory_size,
                                                   glimpse_data_shape=(scale_num, 3) + glimpse_handler.output_shape,
                                                   cf_handler=cf_handler,
                                                   scoremap_processor=scoremap_processor,
                                                   parse_all=verbose_sym_out)
            parsed_outputs_list.append(parsed_outputs)
            #print parsed_outputs['pred_rois']
            #print data_rois_ndarray.asnumpy()[0]
            scores = compute_tracking_score(pred_rois=parsed_outputs['pred_rois'],
                                            truth_rois=data_rois_ndarray.asnumpy()[0],
                                            thresholds=thresholds,
                                            failure_penalty=failure_penalty,
                                            level_reward=level_reward)
            # avg_scores += scores
            backward_inputs = get_backward_input(init_shapes=tracker_init_shapes,
                                                 scores=scores,
                                                 baselines=baselines,
                                                 total_timesteps=BPTT_length,
                                                 attention_steps=attention_handler.total_steps)
            if 'train' == args.mode:
                tracker.backward(**backward_inputs)
                #for k, v in tracker.params_grad.items():
                #    print k, numpy.abs(v.asnumpy()).sum()
                tracker.update(updater=updater)
            # for k, v in tracker.params_grad.items():
            #     if 0 == episode:
            #         accumulative_grad[k][:] = v / float(roll_out_num)
            #     else:
            #         accumulative_grad[k][:] += v / float(roll_out_num)
            if args.visualization:
                data_img_npy = (data_images_ndarray + tracking_iterator.img_mean(data_images_ndarray.shape)).asnumpy()
                p = tracker.params['ScoreMapProcessor:scale0:conv1_weight'].asnumpy()
                visualize_weights(p[0,:,:,:],
                                  win_name="scale0:conv1_weight")
                p = tracker.params['ScoreMapProcessor:scale1:conv1_weight'].asnumpy()
                visualize_weights(p[0,:,:,:],
                                  win_name="scale1:conv1_weight")
                p = tracker.params['ScoreMapProcessor:scale0:conv2_weight'].asnumpy()
                visualize_weights(p[0,:,:,:],
                                  win_name="scale0:conv2_weight")
                p = tracker.params['ScoreMapProcessor:scale1:conv2_weight'].asnumpy()
                visualize_weights(p[0,:,:,:],
                                  win_name="scale1:conv2_weight")
                for i in range(BPTT_length):
                    # for j in range(attention_steps):
                    #    draw_track_res(data_img_npy[0, i, :, :, :], parsed_outputs_list[0]['search_rois'][i, j],
                    #               delay=10, color=(0, 0, 255))
                    draw_track_res(data_img_npy[0, i, :, :, :], parsed_outputs_list[0]['pred_rois'][i],
                                   color=(255, 0, 0))
                    if verbose_sym_out:
                        for j in range(attention_steps):
                            for s in range(scale_num):
                                visualize_weights(parsed_outputs['attention_scoremap'][i, j, s],
                                                  win_name="Attention Scoremap")
                                visualize_weights(parsed_outputs['processed_scoremap'][i, j],
                                                  win_name="Processed Scoremap")
                                visualize_weights(parsed_outputs['pred_glimpse_data'][i],
                                                  win_name="Prediction Glimpse")
                                cv2.waitKey(1)
                    else:
                        cv2.waitKey(1)
                    # print 'k:', k
                    # ch = raw_input()
                    # print v.shape
                    # if 'pred_glimpse_data' == k:
                    #     for i in range(BPTT_length):
                    #         visualize_weights(v[i])
                    # print 'v:', v
                    # ch = raw_input()
            # avg_scores /= roll_out_num
            q_estimation = numpy.cumsum(scores[::-1], axis=0)[::-1]
            baselines[:] -= args.baseline_lr * (baselines - q_estimation)
            #print 'Avg Scores:', avg_scores
            logging.info('Epoch:%d, Iter:%d, Step:%d, Baselines:%s, Read:%g/%s, Write:%g/%s' %
                         (epoch, iter, bptt_step, str(baselines),
                          entropy(parsed_outputs_list[0]['read_controls_prob'].T).mean(),
                          str(parsed_outputs_list[0]['read_controls_prob'].argmax(axis=2)),
                          entropy(parsed_outputs_list[0]['write_controls_prob'].T).mean(),
                          str(parsed_outputs_list[0]['write_controls_prob'].argmax(axis=1))))
    tracker.save_params(dir_path=save_dir, epoch=epoch)