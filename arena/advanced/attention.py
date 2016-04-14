import mxnet as mx


'''
pyramid_glimpse: Generate a spatial pyramid of glimpse sectors, padding zero if necessary
'''
def pyramid_glimpse(data, roi, depth, scale, output_shape, name):
    l = []
    curr_scale = 1.0
    if type(roi) is tuple: # If roi is a tuple then the roi = (center, size)
        roi = mx.symbol.Concat(*roi, num_args=depth)
    for i in range(depth):
        l.append(mx.symbol.SpatialGlimpse(data=data, roi=roi,
                                          output_shape=output_shape,
                                          scale=curr_scale, name="%s-scale%g" %(name, curr_scale)))
        curr_scale *= scale
    ret = mx.symbol.Concat(*l, num_args=depth, name="%s-concat" %name)
    return ret

