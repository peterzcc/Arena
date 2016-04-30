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

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)


sample_length = 11
scale_num = 3
memory_size = 4
attention_steps = 3
image_size = (360, 480)
ctx = mx.gpu()
memory_lstm_props = [LSTMLayerProp(num_hidden=128, dropout=0.),
                     LSTMLayerProp(num_hidden=128, dropout=0.)]
attention_lstm_props = [LSTMLayerProp(num_hidden=256, dropout=0.),
                        LSTMLayerProp(num_hidden=256, dropout=0.)]

tracking_iterator = TrackingIterator(
    'D:\\HKUST\\2-2\\learning-to-track\\datasets\\OTB100-processed\\otb100-video.lst',
    output_size=image_size,
    resize=True)
glimpse_handler = GlimpseHandler(scale_mult=1.8, scale_num=scale_num, output_shape=(133, 133))
perception_handler = PerceptionHandler(net_type='VGG-M')
cf_handler = CorrelationFilterHandler(rows=64, cols=64, gaussian_sigma_factor=10, regularizer=0.01,
                                      perception_handler=perception_handler,
                                      glimpse_handler=glimpse_handler)
scoremap_processor = ScoreMapProcessor(dim_in=(96, 64, 64), num_filter=2, scale_num=scale_num)
memory_handler = MemoryHandler(cf_handler=cf_handler, scoremap_processor=scoremap_processor,
                               memory_size=memory_size,
                               lstm_layer_props=memory_lstm_props)
attention_handler = AttentionHandler(glimpse_handler=glimpse_handler, cf_handler=cf_handler,
                                     scoremap_processor=scoremap_processor,
                                     total_steps=attention_steps,
                                     lstm_layer_props=attention_lstm_props,
                                     fixed_variance=True)

###########################  1st: Build the symbolic computation logic #############################

data_images = mx.symbol.Variable('data_images')
data_rois = mx.symbol.Variable('data_rois')
init_write_control_flag = mx.symbol.Variable('init_write_control_flag')
update_factor = mx.symbol.Variable('update_factor')
roi_var = mx.symbol.Variable('roi_var')

data_rois = mx.symbol.SliceChannel(data_rois, num_outputs=2, axis=2)
data_images = mx.symbol.SliceChannel(data_images, num_outputs=sample_length, axis=1,
                                     squeeze_axis=True)
data_centers = mx.symbol.SliceChannel(data_rois[0], num_outputs=sample_length, axis=1,
                                      squeeze_axis=True)
data_sizes = mx.symbol.SliceChannel(data_rois[1], num_outputs=sample_length, axis=1,
                                    squeeze_axis=True)

init_shapes = OrderedDict()
sym_out = OrderedDict()


# 0. Initialize the parameters in the handler

init_memory, init_memory_data, init_memory_data_shapes = memory_handler.init_memory(ctx=ctx)
init_shapes.update(init_memory_data_shapes)
init_attention_lstm_data, init_attention_lstm_shape = attention_handler.init_lstm(ctx=ctx)
init_shapes.update(init_attention_lstm_shape)

# 1. Get template from the first frame and insert it into the memory
template = cf_handler.get_multiscale_template(img=data_images[0], center=data_centers[0],
                                              size=data_sizes[0], postfix='_t0')
memory, write_sym_out, write_init_shapes = memory_handler.write(memory=init_memory,
                                                                update_multiscale_template=template,
                                                                control_flag=init_write_control_flag,
                                                                update_factor=update_factor,
                                                                timestamp=0)
sym_out.update(write_sym_out)
init_shapes.update(write_init_shapes)


# 2. Track following the Perceive, Attend and Memorize procedure

init_center = data_centers[0]
init_size = data_sizes[0]
tracking_state = None
counter_history = OrderedDict()
visiting_timestamp_history = OrderedDict()
read_template = OrderedDict()

for i in range(1, sample_length):
    if i > 1:
        # 2.1 Read template from the memory
        memory, template, read_sym_out, read_init_shapes = \
            memory_handler.read(memory=memory, img=data_images[i], center=init_center,
                                size=init_size, timestamp=i)
        sym_out.update(read_sym_out)
        init_shapes.update(read_init_shapes)
        counter_history['counter:read_t%i' %i] = memory.status.counter
        visiting_timestamp_history['visiting_timestamp:read_t%i' %i] = memory.status.visiting_timestamp
        read_template['numerators:read_t%i' %i] = template.numerator
        read_template['denominators:read_t%i' % i] = template.denominator
    # 2.2 Attend
    tracking_states, init_center, init_size, pred_center, pred_size, attend_sym_out, \
    attend_init_shapes = attention_handler.attend(
        img=data_images[i], init_center=init_center, init_size=init_size,
        multiscale_template=template, memory=memory,
        ground_truth_roi=mx.symbol.Concat(data_centers[i], data_sizes[i], num_args=2, dim=1),
        timestamp=i, roi_var=roi_var)
    tracking_state = mx.symbol.Concat(*[state.h for state in tracking_states],
                                      num_args=len(tracking_states), dim=1)
    sym_out.update(attend_sym_out)
    init_shapes.update(attend_init_shapes)

    # 2.3 Memorize
    if i < sample_length - 1:
        template = cf_handler.get_multiscale_template(img=data_images[i], center=pred_center,
                                                      size=pred_size, postfix='_t%d_memorize' %i)
        memory, write_sym_out, write_init_shapes = memory_handler.write(memory=memory,
                                                                        tracking_state=tracking_state,
                                                                        update_multiscale_template=template,
                                                                        update_factor=update_factor,
                                                                        timestamp=i)
        sym_out.update(write_sym_out)
        init_shapes.update(write_init_shapes)
        counter_history['counter:write_t%i' % i] = memory.status.counter
        visiting_timestamp_history['visiting_timestamp:write_t%i' % i] = memory.status.visiting_timestamp

############################# 2nd: Build the network ###############################################

data_shapes = OrderedDict([
    ('data_images', (1, sample_length, 3) + image_size),
    ('data_rois', (1, sample_length, 4)),
    ('init_write_control_flag', (1,)),
    ('update_factor', (1,)),
    ('roi_var', (1, 4))])
data_shapes.update(init_shapes)

net = Base(sym=mx.symbol.Group(sym_out.values() +
                               counter_history.values() +
                               visiting_timestamp_history.values() +
                               read_template.values()),
           data_shapes=data_shapes)
net.print_stat()

perception_handler.set_params(net.params)

constant_inputs = OrderedDict()
constant_inputs['init_write_control_flag'] = 2
constant_inputs['update_factor'] = 0.2
constant_inputs["roi_var"] = nd.array(numpy.array([[1E-3, 1E-3, 2E-4, 2E-4]]), ctx=ctx)
constant_inputs.update(init_memory_data)
constant_inputs.update(init_attention_lstm_data)


additional_inputs = OrderedDict()

optimizer = mx.optimizer.create(name='adam', learning_rate=0.001,
                                clip_gradient=None,
                                rescale_grad=1.0, wd=0.)
updater = mx.optimizer.get_updater(optimizer)
start = time.time()

for i in range(100):
    seq_images, seq_rois = tracking_iterator.sample(length=sample_length)
    additional_inputs["data_images"] = seq_images
    additional_inputs["data_rois"] = seq_rois
    if i == 0:
        outputs = net.forward(**(OrderedDict(additional_inputs.items() + constant_inputs.items())))
    else:
        outputs = net.forward(**additional_inputs)
    net.backward()
    for key, grad in net.params_grad.items():
        print key, (grad.asnumpy().sum()), grad.shape
    ch = raw_input()
    net.update(updater=updater)
    for key, output in zip(sym_out.keys() + counter_history.keys() + visiting_timestamp_history.keys() + read_template.keys(),
                           outputs):
        '''
        print key, output.shape
        if 'numerators' in key or 'denominators' in key:
            print numpy.abs(output.asnumpy())[0].sum()
            print numpy.abs(output.asnumpy())[1].sum()
            print numpy.abs(output.asnumpy())[2].sum()
            print numpy.abs(output.asnumpy())[3].sum()
        else:
        '''
        print key, output.asnumpy().sum()
end = time.time()
print sample_length / (end - start)