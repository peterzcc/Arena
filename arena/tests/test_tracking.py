import mxnet as mx
import mxnet.ndarray as nd
import numpy
import cv2
from arena import Base
import time
import pyfftw
import sys
from arena.advanced.attention import *
from arena.advanced.tracking import *
from arena.advanced.memory import *
from arena.advanced.recurrent import *
from arena.advanced.common import *
from arena.iterators import TrackingIterator
from arena.helpers.visualization import *
from arena.helpers.tracking import *

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

'''
Function: build_memory_generator
Description: Get the initial template and memory by analyzing the first frame

'''
def build_memory_generator(image_size, glimpse_handler,
                           cf_handler,
                           perception_handler,
                           memory_handler, ctx):
    sym_out = OrderedDict()
    init_shapes = OrderedDict()


    ############################ 0. Define the input symbols #######################################
    # Variable Inputs: init_image, init_roi
    init_image = mx.symbol.Variable('init_image')
    init_roi = mx.symbol.Variable('init_roi')

    # Constant Inputs: init_write_control_flag, update_factor, init_memory
    init_memory, init_memory_data, init_memory_data_shapes = memory_handler.init_memory(ctx=ctx)
    init_shapes.update(init_memory_data_shapes)
    init_write_control_flag = mx.symbol.Variable('init_write_control_flag')
    update_factor = mx.symbol.Variable('update_factor')

    init_center, init_size = get_roi_center_size(init_roi)

    ############################ 1. Get the computation logic ######################################
    # 1.1 Get the initial template
    glimpse = glimpse_handler.pyramid_glimpse(img=init_image,
                                              center=init_center,
                                              size=init_size,
                                              postfix='_t0')
    template = cf_handler.get_multiscale_template(glimpse=glimpse, postfix='_init_t0')
    # 1.2 Write the template into the memory
    memory, write_sym_out, write_init_shapes = memory_handler.write(memory=init_memory,
                                                                    update_multiscale_template=template,
                                                                    control_flag=init_write_control_flag,
                                                                    update_factor=update_factor,
                                                                    timestamp=0)
    sym_out.update(write_sym_out)
    init_shapes.update(write_init_shapes)

    sym_out['init_memory:numerators'] = memory.numerators
    sym_out['init_memory:denominators'] = memory.denominators
    for i, state in enumerate(memory.states):
       sym_out['init_memory:lstm%d_c' %i] = mx.symbol.BlockGrad(state.c)
       sym_out['init_memory:lstm%d_h' % i] = mx.symbol.BlockGrad(state.h)
    sym_out['init_memory:counter'] = memory.status.counter
    sym_out['init_memory:visiting_timestamp'] = memory.status.visiting_timestamp

    ############################ 2. Get the data shapes ############################################
    data_shapes = OrderedDict([
        ('init_image', (1, 3) + image_size),
        ('init_roi', (1, 4)),
        ('init_write_control_flag', (1,)),
        ('update_factor', (1,))])
    data_shapes.update(init_shapes)

    ############################ 3. Build the network and set initial parameters ###################
    memory_generator = Base(sym=mx.symbol.Group(sym_out.values()),
                            data_shapes=data_shapes,
                            name='MemoryGenerator')
    perception_handler.set_params(memory_generator.params)
    constant_inputs = OrderedDict()
    constant_inputs['init_write_control_flag'] = numpy.array(2)
    constant_inputs['update_factor'] = numpy.array(0.2)
    constant_inputs.update(init_memory_data)
    return memory_generator, sym_out, init_shapes, constant_inputs

class TrackerInitializer(mx.initializer.Normal):
    def _init_weight(self, name, arr):
        super(TrackerInitializer, self)._init_weight(name, arr)
        if 'ScoreMapProcessor' in name:
            print name
            mx.random.normal(0.1, self.sigma, out=arr)
        elif 'roi' in name and 'AttentionHanlder' in name:
            mx.random.normal(0, self.sigma/5, out=arr)

    def _init_bias(self, name, arr):
        super(TrackerInitializer, self)._init_bias(name, arr)


def build_tracker(tracking_length,
                  image_size,
                  deterministic,
                  attention_handler,
                  memory_handler,
                  glimpse_handler,
                  cf_handler,
                  perception_handler,
                  ctx):
    sym_out = OrderedDict()
    init_shapes = OrderedDict()

    ############################ 0. Define the input symbols #######################################
    # Variable Inputs: data_images, data_rois, init_search_roi
    data_images = mx.symbol.Variable('data_images')
    data_rois = mx.symbol.Variable('data_rois')
    init_search_roi = mx.symbol.Variable('init_search_roi')
    init_memory, _, init_memory_data_shapes = memory_handler.init_memory(ctx=ctx)
    init_shapes.update(init_memory_data_shapes)

    # Constant Inputs: update_factor, roi_var
    update_factor = mx.symbol.Variable('update_factor')
    roi_var = mx.symbol.Variable('roi_var')

    init_search_center, init_search_size = get_roi_center_size(init_search_roi)
    data_rois = mx.symbol.SliceChannel(data_rois, num_outputs=2, axis=2)
    data_images = mx.symbol.SliceChannel(data_images, num_outputs=tracking_length, axis=1,
                                         squeeze_axis=True)
    data_centers = mx.symbol.SliceChannel(data_rois[0], num_outputs=tracking_length, axis=1,
                                          squeeze_axis=True)
    data_sizes = mx.symbol.SliceChannel(data_rois[1], num_outputs=tracking_length, axis=1,
                                        squeeze_axis=True)

    init_attention_lstm_data, init_attention_lstm_shape = attention_handler.init_lstm(ctx=ctx)
    init_shapes.update(init_attention_lstm_shape)

    ############################ 1. Get the computation logic ######################################
    memory = init_memory
    memory_status_history = OrderedDict()
    glimpse_history = OrderedDict()
    read_template_history = OrderedDict()
    last_step_memory = OrderedDict()

    for timestamp in range(tracking_length):
        init_glimpse = glimpse_handler.pyramid_glimpse(img=data_images[timestamp],
                                                       center=init_search_center,
                                                       size=init_search_size,
                                                       postfix='_init_t%d' % timestamp)
        glimpse_history['glimpse_init_t%d:center' % timestamp] = init_glimpse.center
        glimpse_history['glimpse_init_t%d:size' % timestamp] = init_glimpse.size
        glimpse_history['glimpse_init_t%d:data' % timestamp] = init_glimpse.data

        # 2.1 Read template from the memory
        memory, template, read_sym_out, read_init_shapes = \
            memory_handler.read(memory=memory, glimpse=init_glimpse, timestamp=timestamp)
        sym_out.update(read_sym_out)
        init_shapes.update(read_init_shapes)
        memory_status_history['counter_after_read_t%d' % timestamp] = memory.status.counter
        memory_status_history[
            'visiting_timestamp_after_read_t%d' % timestamp] = memory.status.visiting_timestamp
        read_template_history['numerators_after_read_t%d' % timestamp] = template.numerator
        read_template_history['denominators_after_read_t%d' % timestamp] = template.denominator

        # 2.2 Attend
        tracking_states, init_search_center, init_search_size, pred_center, pred_size, attend_sym_out, \
        attend_init_shapes = attention_handler.attend(
            img=data_images[timestamp], init_glimpse=init_glimpse,
            multiscale_template=template, memory=memory,
            ground_truth_roi=mx.symbol.Concat(data_centers[timestamp], data_sizes[timestamp],
                                              num_args=2, dim=1),
            timestamp=timestamp, roi_var=roi_var,
            deterministic=deterministic)
        tracking_state = mx.symbol.Concat(*[state.h for state in tracking_states],
                                          num_args=len(tracking_states), dim=1)
        sym_out.update(attend_sym_out)
        init_shapes.update(attend_init_shapes)
        pred_glimpse = glimpse_handler.pyramid_glimpse(img=data_images[timestamp],
                                                       center=pred_center,
                                                       size=pred_size,
                                                       postfix='_pred_t%d' % timestamp)
        glimpse_history['glimpse_next_step_search_t%d:center' % timestamp] = init_search_center
        glimpse_history['glimpse_next_step_search_t%d:size' % timestamp] = init_search_size
        glimpse_history['glimpse_pred_t%d_center' % timestamp] = pred_glimpse.center
        glimpse_history['glimpse_pred_t%d_size' % timestamp] = pred_glimpse.size
        glimpse_history['glimpse_pred_t%d_data' % timestamp] = pred_glimpse.data

        # 2.3 Memorize
        template = cf_handler.get_multiscale_template(glimpse=pred_glimpse,
                                                      postfix='_t%d_memorize' % timestamp)
        memory, write_sym_out, write_init_shapes = memory_handler.write(memory=memory,
                                                                        tracking_state=tracking_state,
                                                                        update_multiscale_template=template,
                                                                        update_factor=update_factor,
                                                                        timestamp=timestamp)
        sym_out.update(write_sym_out)
        init_shapes.update(write_init_shapes)
        if timestamp < tracking_length - 1:
            memory_status_history['counter_after_write_t%d' % timestamp] = memory.status.counter
            memory_status_history[
                'visiting_timestamp_after_write_t%d' % timestamp] = memory.status.visiting_timestamp
        else:
            sym_out.update(write_sym_out)
            last_step_memory['last_step_memory:numerators'] = mx.symbol.BlockGrad(memory.numerators)
            last_step_memory['last_step_memory:denominators'] = mx.symbol.BlockGrad(
                memory.denominators)
            for i, state in enumerate(memory.states):
                last_step_memory['last_step_memory:lstm%d_c' % i] = mx.symbol.BlockGrad(state.c)
                last_step_memory['last_step_memory:lstm%d_h' % i] = mx.symbol.BlockGrad(state.h)
            last_step_memory['last_step_memory:counter'] = mx.symbol.BlockGrad(
                memory.status.counter)
            last_step_memory['last_step_memory:visiting_timestamp'] = mx.symbol.BlockGrad(
                memory.status.visiting_timestamp)
    sym_out.update(memory_status_history)
    sym_out.update(glimpse_history)
    sym_out.update(last_step_memory)

    data_shapes = OrderedDict([
        ('data_images', (1, tracking_length, 3) + image_size),
        ('data_rois', (1, tracking_length, 4)),
        ('update_factor', (1,)),
        ('init_search_roi', (1, 4)),
        ('roi_var', (1, 4))])
    data_shapes.update(init_shapes)
    tracker = Base(sym=mx.symbol.Group(sym_out.values()),
                   data_shapes=data_shapes,
                   initializer=TrackerInitializer(sigma=0.01),
                   name='Tracker')
    perception_handler.set_params(tracker.params)

    constant_inputs = OrderedDict()
    constant_inputs['update_factor'] = 0.1
    constant_inputs["roi_var"] = nd.array(numpy.array([[1E-4, 1E-4, 1E-6, 1E-6]]), ctx=ctx)
    constant_inputs.update(init_attention_lstm_data)
    return tracker, sym_out, init_shapes, constant_inputs

def get_tracker_pred_rois(outputs, sym_out, total_timesteps):
    pred_rois = numpy.zeros((total_timesteps, 4), dtype=numpy.float32)
    center_counter = 0
    size_counter = 0
    for i, (key, output) in enumerate(zip(sym_out.keys(), outputs)):
        if 'glimpse_pred' in key:
            timestamp = get_timestamp(key)
            if 'center' in key:
                pred_rois[timestamp, 0:2] = output.asnumpy()[:]
                center_counter += 1
            elif 'size' in key:
                pred_rois[timestamp, 2:4] = output.asnumpy()[:]
                size_counter += 1
    assert (center_counter == total_timesteps) and (size_counter == total_timesteps)
    return pred_rois


def compute_tracking_score(pred_rois, truth_rois, thresholds=(0.5, 0.7, 0.8, 0.9),
                           failure_penalty=-10, level_reward=1):
    assert pred_rois.shape == truth_rois.shape
    assert len(thresholds) > 1
    overlapping_ratios = cal_rect_int(pred_rois, truth_rois)
    thresholds = sorted(thresholds)
    scores = (failure_penalty * (overlapping_ratios < thresholds[0])).astype(numpy.float32)
    for i, threshold in enumerate(thresholds):
        scores += (level_reward * (overlapping_ratios >= threshold)).astype(numpy.float32)
    return scores


def get_memory_controls(outputs, sym_out, total_timesteps):
    read_controls = numpy.empty((total_timesteps, ), dtype=numpy.float32)
    write_controls = numpy.empty((total_timesteps, ), dtype=numpy.float32)

    for i, (key, output) in enumerate(zip(sym_out.keys(), outputs)):
        if 'read:chosen_ind' in key and 'action' in key:
            timestamp = get_timestamp(key)
            read_controls[timestamp] = output.asnumpy()
        elif 'write:control_flag' in key and 'action' in key:
            timestamp = get_timestamp(key)
            write_controls[timestamp] = output.asnumpy()
    return read_controls, write_controls


def get_backward_input(init_shapes, scores, baselines, total_timesteps, attention_steps):
    assert scores.shape == baselines.shape
    assert 1 == len(scores.shape)
    assert scores.shape[0] == total_timesteps
    backward_inputs = OrderedDict()
    scores = numpy.cumsum(scores[::-1], axis=0)[::-1]
    advantages = (scores - baselines)/20000.0
    counter_checking = {'search_roi':0,
                        'pred_roi':0,
                        'init_roi':0,
                        'read:chosen_ind':0,
                        'write:control_flag':0}
    counter_ground_truth = {'search_roi':total_timesteps * (attention_steps - 1),
                            'pred_roi':total_timesteps,
                            'init_roi':total_timesteps,
                            'read:chosen_ind':total_timesteps,
                            'write:control_flag':total_timesteps}
    for key in init_shapes.keys():
        if 'score' in key:
            timestamp = get_timestamp(key)
            if 'search_roi' in key:
                backward_inputs[key] = advantages[timestamp]
                counter_checking['search_roi'] += 1
            elif 'pred_roi' in key:
                backward_inputs[key] = advantages[timestamp]
                counter_checking['pred_roi'] += 1
            elif 'init_roi' in key:
                if timestamp < total_timesteps - 1:
                    backward_inputs[key] = advantages[timestamp + 1]
                else:
                    backward_inputs[key] = 0
                counter_checking['init_roi'] += 1
            elif 'read:chosen_ind' in key:
                backward_inputs[key] = advantages[timestamp] * 100
                counter_checking['read:chosen_ind'] += 1
            elif 'write:control_flag' in key:
                if timestamp < total_timesteps - 1:
                    backward_inputs[key] = advantages[timestamp + 1] * 100
                else:
                    backward_inputs[key] = 0
                counter_checking['write:control_flag'] += 1
            else:
                raise NotImplementedError, 'Only support %s, find key="%s"' \
                                           %(str(counter_checking.keys()), key)
    assert counter_checking == counter_ground_truth, counter_checking
    return backward_inputs


sample_length = 16
BPTT_length = 15

thresholds=(0.5, 0.7, 0.8, 0.9)
failure_penalty = -10
level_reward = 1

scale_num = 3
memory_size = 3
attention_steps = 3
image_size = (360, 480)
ctx = mx.gpu()
memory_lstm_props = [LSTMLayerProp(num_hidden=128, dropout=0.),
                     LSTMLayerProp(num_hidden=128, dropout=0.)]
attention_lstm_props = [LSTMLayerProp(num_hidden=128, dropout=0.),
                        LSTMLayerProp(num_hidden=128, dropout=0.)]

tracking_iterator = TrackingIterator(
    'D:\\HKUST\\2-2\\learning-to-track\\datasets\\OTB100-processed\\otb100-video.lst',
    output_size=image_size,
    resize=True)
glimpse_handler = GlimpseHandler(scale_mult=1.8, scale_num=scale_num, output_shape=(133, 133))
perception_handler = PerceptionHandler(net_type='VGG-M')
cf_handler = CorrelationFilterHandler(rows=64, cols=64, gaussian_sigma_factor=10, regularizer=0.01,
                                      perception_handler=perception_handler,
                                      glimpse_handler=glimpse_handler)
scoremap_processor = ScoreMapProcessor(dim_in=(96, 64, 64), num_filter=4, scale_num=scale_num)
memory_handler = MemoryHandler(cf_handler=cf_handler, scoremap_processor=scoremap_processor,
                               memory_size=memory_size,
                               lstm_layer_props=memory_lstm_props)
attention_handler = AttentionHandler(glimpse_handler=glimpse_handler, cf_handler=cf_handler,
                                     scoremap_processor=scoremap_processor,
                                     total_steps=attention_steps,
                                     lstm_layer_props=attention_lstm_props,
                                     fixed_variance=True)

# 1. Build the memory generator that initialze the memory by analyzing the first frame


memory_generator, mem_sym_out, mem_init_shapes, mem_constant_inputs= \
    build_memory_generator(image_size=image_size,
                           memory_handler=memory_handler,
                           glimpse_handler=glimpse_handler,
                           cf_handler=cf_handler,
                           perception_handler=perception_handler,
                           ctx=ctx)
memory_generator.print_stat()

# 2. Build the tracker following the Perceive, Attend and Memorize procedure
tracker, tracker_sym_out, tracker_init_shapes, tracker_constant_inputs= \
    build_tracker(image_size=image_size,
                  tracking_length=BPTT_length,
                  deterministic=False,
                  memory_handler=memory_handler,
                  glimpse_handler=glimpse_handler,
                  cf_handler=cf_handler,
                  attention_handler=attention_handler,
                  perception_handler=perception_handler,
                  ctx=ctx)
tracker.print_stat()


baselines = numpy.zeros((BPTT_length,), dtype=numpy.float32)
optimizer = mx.optimizer.create(name='adam', learning_rate=0.001,
                                clip_gradient=None,
                                rescale_grad=1.0, wd=0.00001)
updater = mx.optimizer.get_updater(optimizer)
start = time.time()
for iter in range(100000):
    seq_images, seq_rois = tracking_iterator.sample(length=sample_length, interval_step=1)
    #print seq_images.shape
    #print seq_rois.shape
    init_image_ndarray = seq_images[:1].reshape((1,) + seq_images.shape[1:])
    init_roi_ndarray = seq_rois[:1]
    #print init_roi_ndarray.shape
    #print init_image_ndarray.shape
    additional_inputs = OrderedDict()
    additional_inputs['init_image'] = init_image_ndarray
    additional_inputs['init_roi'] = init_roi_ndarray
    #for k, v in mem_constant_inputs.items():
    #    print k, v.shape
    if 0 == iter:
        mem_outputs = memory_generator.forward(is_train=False,
                                               **(OrderedDict(additional_inputs.items() +
                                                              mem_constant_inputs.items())))
    else:
        mem_outputs = memory_generator.forward(is_train=False, **additional_inputs)

    data_images_ndarray = seq_images[1:(BPTT_length+1)].reshape((1, BPTT_length,) + seq_images.shape[1:])
    data_rois_ndarray = seq_rois[1:(BPTT_length+1)].reshape((1, BPTT_length, 4))
    additional_inputs = OrderedDict()
    additional_inputs['data_images'] = data_images_ndarray
    additional_inputs['data_rois'] = data_rois_ndarray
    additional_inputs['init_search_roi'] = init_roi_ndarray
    for i, k in enumerate(mem_sym_out.keys()):
        if 'init_memory' in k:
            additional_inputs[k] = mem_outputs[i]
    if iter == 0:
        tracker_outputs = tracker.forward(is_train=True, **(OrderedDict(additional_inputs.items() +
                                                tracker_constant_inputs.items())))
    else:
        tracker_outputs = tracker.forward(is_train=True, **additional_inputs)
    read_controls, write_controls = get_memory_controls(tracker_outputs, tracker_sym_out, BPTT_length)
    pred_rois = get_tracker_pred_rois(tracker_outputs, tracker_sym_out, BPTT_length)
    print pred_rois
    print data_rois_ndarray.asnumpy()[0]
    scores = compute_tracking_score(pred_rois=pred_rois,
                                    truth_rois=data_rois_ndarray.asnumpy()[0],
                                    thresholds=thresholds,
                                    failure_penalty=failure_penalty,
                                    level_reward=level_reward)
    print 'Scores:', scores
    print 'Baseline:', baselines
    print 'Read Controls:', read_controls
    print 'Write Controls:', write_controls
    backward_inputs = get_backward_input(init_shapes=tracker_init_shapes,
                                         scores=scores,
                                         baselines=baselines,
                                         total_timesteps=BPTT_length,
                                         attention_steps=attention_handler.total_steps)
    tracker.backward(**backward_inputs)
    q_estimation = numpy.cumsum(scores[::-1], axis=0)[::-1]
    baselines[:] -= 0.0001 * (baselines - q_estimation)
    for k, v in tracker.params_grad.items():
        print k, numpy.abs(v.asnumpy()).sum()
    #ch = raw_input()
    tracker.update(updater=updater)
end = time.time()
print sample_length / (end - start)