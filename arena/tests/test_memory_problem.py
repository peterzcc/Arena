import logging
import mxnet as mx
import numpy
import win32process
from   mxnet.executor_manager import _bind_exec
from   mxnet.ndarray import array

def get_symbol( nDepth = 3):
    data        = mx.symbol.Variable( 'data', shape = ( 40, 28, 28 ) )
    for i in xrange( nDepth - 1 ):
        data    = mx.symbol.Convolution( data = data, kernel = ( 3, 3 ), pad = ( 2, 2 ),
                                         num_filter = 128 )
        data    = mx.symbol.Activation( data = data, act_type = 'relu' )
    collapsed   = mx.symbol.Convolution( data = data, kernel = ( 1, 1 ), num_filter = 1 )
    flatten     = mx.symbol.Flatten( data = collapsed )
    softmax     = mx.symbol.SoftmaxOutput( data = flatten, name = 'softmax' )
    return softmax

def run():
    batch_size  = 32
    nPlane      = 40
    nDepth      = 4
    #nDepth      = 3
    width       = 28
    height      = 28
    symbol      = get_symbol( nDepth )

    param_names = [ name for name in symbol.list_arguments() if name not in [ 'data', 'softmax_label' ] ]
    data_shapes = { 'data'          : (batch_size, nPlane, width, height),
                    'softmax_label' : (batch_size,)}
    train_exec  = _bind_exec( symbol, mx.cpu(), data_shapes, param_names,
                             need_grad=True, base_exec=None,
                             shared_data_arrays={})

    logging.getLogger().setLevel( logging.INFO )
    for i in xrange( 10000 ):
        data_source     = array( numpy.random.binomial(1,0.5, batch_size * nPlane * width * height ).reshape( batch_size, nPlane, width, height ) )
        label_source    = array( numpy.random.choice( width * height, batch_size ) )
        data_source.copyto( train_exec.arg_dict[ 'data' ] )
        label_source.copyto( train_exec.arg_dict[ 'softmax_label' ] )
        train_exec.forward(is_train=True)
        memUsed         = float( win32process.GetProcessMemoryInfo(-1)[ 'PagefileUsage' ] ) / ( 1024. * 1024. )
        logging.info( 'Memory used: %s MB' % memUsed )

if __name__ == '__main__':
    run()