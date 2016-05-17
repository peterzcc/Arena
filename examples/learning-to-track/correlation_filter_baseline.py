from arena.advanced.attention import *
from arena.advanced.tracking import *
from arena.advanced.memory import *
from arena.advanced.recurrent import *
from arena.advanced.common import *
from arena import Base
from arena.iterators import TrackingIterator
from arena.helpers.visualization import *
import numpy


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

def build_template_extraction_net(glimpse_handler, perception_handler, cf_handler):
    image = mx.symbol.Variable('image')
    roi = mx.symbol.Variable('roi')
    center, size = get_roi_center_size(roi)
    glimpse = glimpse_handler.pyramid_glimpse(img=image, center=center, size=size, postfix='_t0')
    template = cf_handler.get_multiscale_template(glimpse=glimpse, postfix='_init_t0')
    data_shapes = {'image': (1, 3) + image_size, 'roi': (1, 4)}
    template_extraction_net = Base(sym=mx.symbol.Group([template.numerator, template.denominator]),
                                   data_shapes=data_shapes)
    perception_handler.set_params(template_extraction_net.params)
    return template_extraction_net

def build_scoremap_computation_net(glimpse_handler, perception_handler, cf_handler, scoremap_processor):
    image = mx.symbol.Variable('image')
    roi = mx.symbol.Variable('roi')
    center, size = get_roi_center_size(roi)
    glimpse = glimpse_handler.pyramid_glimpse(img=image, center=center, size=size, postfix='_t0')
    numerator = mx.symbol.Variable('numerator')
    denominator = mx.symbol.Variable('denominator')
    scoremap = cf_handler.get_multiscale_scoremap(multiscale_template=ScaleCFTemplate(numerator=numerator,
                                                                                      denominator=denominator),
                                                  glimpse=glimpse)
    scoremap = scoremap_processor.scoremap_processing(multiscale_scoremap=scoremap, postfix="_t0")
    data_shapes = {'image': (1, 3) + image_size, 'roi': (1, 4), 'numerator': cf_handler.numerator_shape,
                   'denominator': cf_handler.denominator_shape}
    scoremap_computation_net = Base(sym=scoremap, data_shapes=data_shapes, initializer=TrackerInitializer(sigma=0.01))
    perception_handler.set_params(scoremap_computation_net.params)
    return scoremap_computation_net

scale_mult = 1.0
scale_num = 1
init_scale = 2.0

cf_gaussian_sigma_factor = 10
cf_regularizer = 0.01

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
                                       num_filter=4,
                                       scale_num=scale_num)

image_size = (480, 540)
sample_length = 200
tracking_iterator = TrackingIterator('D:\\HKUST\\2-2\\learning-to-track\\datasets\\OTB100-processed\\temp.lst',
                                     output_size=image_size, resize=True)

template_extraction_net = build_template_extraction_net(glimpse_handler=glimpse_handler,
                                                        perception_handler=perception_handler,
                                                        cf_handler=cf_handler)
scoremap_computation_net = build_scoremap_computation_net(glimpse_handler=glimpse_handler,
                                                          perception_handler=perception_handler,
                                                          scoremap_processor=scoremap_processor,
                                                          cf_handler=cf_handler)
seq_images, seq_rois = tracking_iterator.sample(length=sample_length, interval_step=1, verbose=False,
                                                random_perturbation_noise=0)
init_image_ndarray = seq_images[:1].reshape((1,) + seq_images.shape[1:])
init_roi_ndarray = seq_rois[:1]
outputs = template_extraction_net.forward(image=init_image_ndarray, roi=init_roi_ndarray)
template = ScaleCFTemplate(numerator=outputs[0].asnumpy(), denominator=outputs[1].asnumpy())
draw_track_res(im=init_image_ndarray.asnumpy()[0], roi=init_roi_ndarray.asnumpy()[0], win_name="T2")
cv2.waitKey()

for t in range(1, sample_length):
    image_ndarray = seq_images[t:(t+1)].reshape((1,) + seq_images.shape[1:])
    data_img_npy = (image_ndarray + tracking_iterator.img_mean(image_ndarray.shape)).asnumpy()
    outputs = scoremap_computation_net.forward(image=image_ndarray, roi=init_roi_ndarray,
                                               numerator=template.numerator, denominator=template.denominator)
    #scoremap = outputs[0].asnumpy()[0].sum(axis=0)
    #scoremap = scoremap/scoremap.max()
    scoremap = outputs[0].asnumpy()[0,0]
    print scoremap.shape
    indx = scoremap.argmax()
    coordinates = numpy.unravel_index(indx, scoremap.shape)
    dx = coordinates[1] / float(scoremap.shape[1]) - 0.5
    dy = coordinates[0] / float(scoremap.shape[0]) - 0.5
    old_cx, old_cy, old_sx, old_sy = init_roi_ndarray.asnumpy()[0].tolist()
    new_cx = old_cx + dx * old_sx *scale_mult
    new_cy = old_cy + dy * old_sy *scale_mult
    print new_cx, new_cy
    init_roi_ndarray = nd.array([[new_cx, new_cy, old_sx, old_sy]])
    template_outputs = template_extraction_net.forward(image=image_ndarray, roi=init_roi_ndarray)
    alpha = 0.003
    template = ScaleCFTemplate(numerator=(1-alpha)*template.numerator + alpha*template_outputs[0].asnumpy(),
                               denominator=(1-alpha)*template.denominator + alpha*template_outputs[1].asnumpy())
    visualize_weights(outputs[0].asnumpy().sum(axis=1))
    draw_track_res(im=data_img_npy[0], roi=init_roi_ndarray.asnumpy()[0])
    draw_track_res(im=data_img_npy[0], roi=seq_rois[t:(t+1)].asnumpy()[0], win_name="T2")
    cv2.waitKey()

