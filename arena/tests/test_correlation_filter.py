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

sample_length = 2
scale_num = 3
memory_size = 4
attention_steps = 1
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
scoremap_processor = ScoreMapProcessor(dim_in=(96, 64, 64), num_filter=8, scale_num=scale_num)
memory_handler = MemoryHandler(cf_handler=cf_handler, scoremap_processor=scoremap_processor,
                               memory_size=memory_size,
                               lstm_layer_props=memory_lstm_props)
attention_handler = AttentionHandler(glimpse_handler=glimpse_handler, cf_handler=cf_handler,
                                     scoremap_processor=scoremap_processor,
                                     total_steps=attention_steps,
                                     lstm_layer_props=attention_lstm_props,
                                     fixed_variance=True)

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



init_glimpse = glimpse_handler.pyramid_glimpse(img=data_images[0], center=data_centers[0],
                                               size=data_sizes[0], timestamp=0)
init_template = cf_handler.get_multiscale_template(glimpse=init_glimpse, postfix='_t0')
init_memory, init_memory_data, init_memory_data_shapes = memory_handler.init_memory(ctx=ctx)
init_shapes.update(init_memory_data_shapes)

init_attention_lstm_data, init_attention_lstm_shape = attention_handler.init_lstm(ctx=ctx)
init_shapes.update(init_attention_lstm_shape)

# init_read_chosen_ind = mx.symbol.Variable('init_read_chosen_ind')


memory, write_sym_out, write_init_shapes = memory_handler.write(memory=init_memory,
                                                                update_multiscale_template=init_template,
                                                                control_flag=init_write_control_flag,
                                                                update_factor=update_factor,
                                                                timestamp=0)
sym_out.update(write_sym_out)
init_shapes.update(write_init_shapes)

print 'write_sym_out', write_sym_out
print 'write_init_shapes', write_init_shapes

init_center = data_centers[0]
init_size = data_sizes[0]
tracking_states, next_step_init_center, next_step_init_size, pred_center, pred_size, attend_sym_out, \
attend_init_shapes = attention_handler.attend(
    img=data_images[1], init_center=init_center, init_size=init_size,
    multiscale_template=init_template, memory=memory,
    ground_truth_roi=mx.symbol.Concat(data_centers[1], data_sizes[1], num_args=2, dim=1),
    timestamp=1, roi_var=roi_var)

sym_out.update(attend_sym_out)
init_shapes.update(attend_init_shapes)

data_shapes = OrderedDict([
    ('data_images', (1, sample_length, 3) + image_size),
    ('data_rois', (1, sample_length, 4)),
    ('init_write_control_flag', (1,)),
    ('update_factor', (1,)),
    ('roi_var', (1, 4))])
data_shapes.update(init_shapes)


data_shapes.pop("init_write_control_flag", None)
data_shapes.pop("update_factor", None)
data_shapes.pop("MemoryHandler:memory_init:numerators", None)
data_shapes.pop("MemoryHandler:memory_init:denominators", None)
data_shapes.pop("MemoryHandler:memory_init:visiting_timestamp", None)
data_shapes.pop("MemoryHandler:memory_init:counter", None)
data_shapes.pop('MemoryHandler:memory_init:lstm0_c', None)
#init_memory_data.pop('MemoryHandler:memory_init:lstm0_c', None)
#data_shapes.pop('MemoryHandler:memory_init:lstm0_h', None)
data_shapes.pop('MemoryHandler:memory_init:lstm1_c', None)
#init_memory_data.pop('MemoryHandler:memory_init:lstm1_c', None)
#data_shapes.pop('MemoryHandler:memory_init:lstm1_h', None)


print 'data_shapes:', data_shapes
print memory_to_sym_dict(memory)
print sym_out

# net = Base(sym=mx.symbol.Group(symbols=block_all(memory_to_sym_dict(memory).values())),
#            data_shapes=data_shapes)
net = Base(sym=mx.symbol.Group(sym_out.values() + [pred_center, pred_size]),
           data_shapes=data_shapes)

net.print_stat()
perception_handler.set_params(net.params)

constant_inputs = OrderedDict()
#constant_inputs['init_write_control_flag'] = 2
# constant_inputs['init_read_chosen_ind'] = 2
#constant_inputs['update_factor'] = 0.2
#constant_inputs.update(init_memory_data)
constant_inputs.update(init_attention_lstm_data)

seq_images, seq_rois = tracking_iterator.sample(length=sample_length)
additional_inputs = OrderedDict()
additional_inputs["data_images"] = seq_images
additional_inputs["data_rois"] = seq_rois
additional_inputs["roi_var"] = nd.array(numpy.array([[1E-3, 1E-3, 2E-4, 2E-4]]), ctx=ctx)
print 'Run'
outputs = net.forward(**(OrderedDict(additional_inputs.items() + constant_inputs.items())))
print outputs
# outputs = net.forward(**additional_inputs)
for key, output in zip(sym_out.keys() + ['pred_center', 'pred_size'], outputs):
    print key, output.shape, output.asnumpy()
    if 'counter' in key or 'visiting_timestamp' in key or 'chosen_ind' in key:
        print output.asnumpy()
    if 'numerators' in key or 'denominators' in key:
        print numpy.abs(output.asnumpy())[0].sum()
        print numpy.abs(output.asnumpy())[1].sum()
        print numpy.abs(output.asnumpy())[2].sum()
        print numpy.abs(output.asnumpy())[3].sum()

'''
net.print_stat()

output = net.forward(**(OrderedDict(additional_inputs.items())))[0]
for i in range(output.asnumpy().shape[1]):
    cv2.imshow("image", output.asnumpy()[2, i] / output.asnumpy()[2, i].max())
    cv2.waitKey()
'''

'''
for i in range(sample_length):
    glimpse = glimpse_handler.pyramid_glimpse(img=data_images[i], center=data_centers[i],
                                              size=data_sizes[i], timestamp=i)
    multiscale_template = cf_handler.get_multiscale_template(glimpse=glimpse, postfix='_t%d' %i)
    scoremap = cf_handler.get_multiscale_scoremap(multiscale_template=multiscale_template,
                                                  glimpse=glimpse, postfix="_t%d" %i)
    multiscale_template_l.append(multiscale_template)
    scoremap_l.append(scoremap)

net_sym = mx.symbol.Group(symbols=scoremap_l)

data_shapes = {'data_images':(1, sample_length, 3) + image_size,
               'data_rois': (1, sample_length, 4)}

net = Base(sym=net_sym, data_shapes=data_shapes)
perception_handler.set_params(net.params)

start = time.time()
for i in range(100):
    seq_images, seq_rois = tracking_iterator.sample(length=sample_length)
    outputs = net.forward(data_shapes={'data_images': seq_images.shape, 'data_rois': seq_rois.shape},
            data_images=seq_images, data_rois=seq_rois)
    for output in outputs:
        print output.asnumpy().shape
        #visualize_weights(output.asnumpy()[0])
        # for i in range(output.asnumpy().shape[1]):
        #     cv2.imshow("image", output.asnumpy()[2, i]/output.asnumpy()[2,i].max())
        #     cv2.waitKey()
end = time.time()
print end-start
'''
