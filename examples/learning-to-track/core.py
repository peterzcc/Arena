import mxnet as mx
import mxnet.ndarray as nd
from arena.advanced.attention import *
from arena.advanced.tracking import *
from arena.advanced.memory import *
from arena.advanced.recurrent import *
from arena.advanced.common import *
from arena import Base

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
        sym_out['init_memory:lstm%d_c' % i] = mx.symbol.BlockGrad(state.c)
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
    constant_inputs['update_factor'] = numpy.array(0.1)
    constant_inputs.update(init_memory_data)
    return memory_generator, sym_out, init_shapes, constant_inputs


class TrackerInitializer(mx.initializer.Normal):
    def _init_weight(self, name, arr):
        super(TrackerInitializer, self)._init_weight(name, arr)
        if 'ScoreMapProcessor' in name:
            mx.random.normal(0.1, self.sigma, out=arr)
        elif 'init_roi' in name and 'AttentionHanlder' in name:
            mx.random.normal(0, self.sigma, out=arr)
        elif 'search_roi' in name and 'AttentionHanlder' in name:
            mx.random.normal(0, self.sigma, out=arr)

    def _init_bias(self, name, arr):
        super(TrackerInitializer, self)._init_bias(name, arr)
        if 'conv' in name:
            arr[:] = 0.1


def build_tracker(tracking_length,
                  image_size,
                  deterministic,
                  attention_handler,
                  memory_handler,
                  glimpse_handler,
                  cf_handler,
                  perception_handler,
                  default_update_factor,
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
    center_var = mx.symbol.Variable('center_var')
    size_var = mx.symbol.Variable('size_var')

    init_search_center, init_search_size = get_roi_center_size(init_search_roi)
    data_rois = mx.symbol.SliceChannel(data_rois, num_outputs=2, axis=2)
    data_images = mx.symbol.SliceChannel(data_images, num_outputs=tracking_length, axis=1,
                                         squeeze_axis=True)
    data_centers = mx.symbol.SliceChannel(data_rois[0], num_outputs=tracking_length, axis=1,
                                          squeeze_axis=True)
    data_sizes = mx.symbol.SliceChannel(data_rois[1], num_outputs=tracking_length, axis=1,
                                        squeeze_axis=True)

    ############################ 1. Get the computation logic ######################################
    memory = init_memory
    memory_status_history = OrderedDict()
    glimpse_history = OrderedDict()
    last_step_memory = OrderedDict()

    for timestamp in range(tracking_length):
        # 2.1 Get the initial glimpse
        glimpse_history['glimpse_init_t%d:center' % timestamp] = mx.symbol.BlockGrad(init_search_center)
        glimpse_history['glimpse_init_t%d:size' % timestamp] = mx.symbol.BlockGrad(init_search_size)

        # 2.2 Attend with the help of memory
        memory, init_search_center, init_search_size, pred_center, pred_size, attend_sym_out, \
        attend_init_shapes = attention_handler.attend(
            img=data_images[timestamp],
            init_search_center=init_search_center,
            init_search_size=init_search_size,
            memory=memory,
            ground_truth_roi=mx.symbol.Concat(data_centers[timestamp], data_sizes[timestamp],
                                              num_args=2, dim=1),
            timestamp=timestamp,
            center_var=center_var,
            size_var=size_var,
            deterministic=deterministic)
        sym_out.update(attend_sym_out)
        init_shapes.update(attend_init_shapes)
        pred_glimpse = glimpse_handler.pyramid_glimpse(img=data_images[timestamp],
                                                       center=pred_center,
                                                       size=pred_size,
                                                       postfix='_pred_t%d' % timestamp)
        glimpse_history['glimpse_next_step_search_t%d:center' % timestamp] = mx.symbol.BlockGrad(init_search_center)
        glimpse_history['glimpse_next_step_search_t%d:size' % timestamp] = mx.symbol.BlockGrad(init_search_size)
        glimpse_history['glimpse_pred_t%d_center' % timestamp] = pred_glimpse.center
        glimpse_history['glimpse_pred_t%d_size' % timestamp] = pred_glimpse.size
        glimpse_history['glimpse_pred_t%d_data' % timestamp] = pred_glimpse.data

        # 2.3 Memorize
        template = cf_handler.get_multiscale_template(glimpse=pred_glimpse,
                                                      postfix='_t%d_memorize' % timestamp)
        memory, write_sym_out, write_init_shapes = memory_handler.write(memory=memory,
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
        ('init_search_roi', (1, 4))])
    if attention_handler.total_steps > 1:
        data_shapes['center_var'] = (1, 2)
        data_shapes['size_var'] = (1, 2)
    data_shapes.update(init_shapes)
    tracker = Base(sym=mx.symbol.Group(sym_out.values()),
                   data_shapes=data_shapes,
                   initializer=TrackerInitializer(sigma=0.01),
                   name='Tracker')
    perception_handler.set_params(tracker.params)

    constant_inputs = OrderedDict()
    constant_inputs['update_factor'] = default_update_factor
    if attention_handler.total_steps > 1:
        constant_inputs["center_var"] = nd.array(numpy.array([[1, 1]]), ctx=ctx)
        constant_inputs["size_var"] = nd.array(numpy.array([[1, 1]]), ctx=ctx)
    return tracker, sym_out, init_shapes, constant_inputs

