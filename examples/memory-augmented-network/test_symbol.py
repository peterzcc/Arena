
###################################### Embedding ######################################

import mxnet as mx
ctx_haha = mx.cpu()

context = mx.symbol.Variable('context')
embed_weight = mx.sym.Variable("embed_weight")

C = mx.nd.array([[1,2,3,4,5],[1,3,5,7,4],[1,2,5,7,8]],ctx=ctx_haha)
E = mx.random.uniform(1,10,shape=(10,3),ctx=ctx_haha)

Ain_c = mx.sym.Embedding(data=context, input_dim=10, weight=embed_weight, output_dim=3, name='Ain_c')
c_exec = Ain_c.bind(ctx=ctx_haha, args={'context' : C, 'embed_weight': E})

c_exec.forward()
result = c_exec.outputs[0].asnumpy()
print result

###################################### Reshape ######################################



###################################### dot ######################################

import mxnet as mx
ctx_haha = mx.gpu(0)

context = mx.symbol.Variable('context')
input = mx.symbol.Variable('input')

C = mx.nd.array([[1,2,3],[1,3,5],[1,2,5]],ctx=ctx_haha)
C.asnumpy()
I = mx.nd.array([[1],[2],[3]],ctx=ctx_haha)
I.asnumpy()

MM = mx.sym.dot(context,input)

mm_exec = MM.bind(ctx=ctx_haha, args={'context' : C, 'input': I})

mm_exec.forward()
result = mm_exec.outputs[0].asnumpy()
print result


###################################### expand_dims ######################################

import mxnet as mx
ctx_haha = mx.cpu()

key = mx.symbol.Variable('key')

k = mx.nd.array([[1,2,3],[1,3,5],[2,4,6],[4,6,8]],ctx=ctx_haha)
k.asnumpy()

MM = mx.sym.expand_dims(key, axis=1)

mm_exec = MM.bind(ctx=ctx_haha, args={'key' : k})

mm_exec.forward()
result = mm_exec.outputs[0].asnumpy()
print result.shape
print result


###################################### broadcast_mul & sum ######################################

import mxnet as mx
ctx_haha = mx.gpu(0)

probs3dim = mx.symbol.Variable('probs3dim')
Bin = mx.symbol.Variable('Bin')

P = mx.nd.array([[[1],[2],[3],[4],[5]],[[2],[4],[6],[8],[10]]],ctx=ctx_haha)
P.asnumpy()
B = mx.nd.array([[[1,2,3],[1,3,5],[1,2,5],[1,2,5],[1,2,5]],[[1,2,3],[1,3,5],[1,2,5],[1,2,5],[1,2,5]]],ctx=ctx_haha)
B.asnumpy()


MM = mx.sym.broadcast_mul(probs3dim, Bin)
MM = mx.sym.sum(MM, axis=1)

mm_exec = MM.bind(ctx=ctx_haha, args={'Bin' : B, 'probs3dim': P})

mm_exec.forward()
result = mm_exec.outputs[0].asnumpy()
print result


###################################### sum ######################################

import mxnet as mx
ctx_haha = mx.cpu()

probs3dim = mx.symbol.Variable('probs3dim')
Bin = mx.symbol.Variable('Bin')

#P = mx.nd.array([[1,2,3,4,5],[2,4,6,8,10]],ctx=ctx_haha)
#P.asnumpy()
B = mx.nd.array([[[1,2,3],[1,3,5],[1,2,5],[1,2,5],[1,2,5]],[[1,2,3],[1,3,5],[1,2,5],[1,2,5],[1,2,5]]],ctx=ctx_haha)
B.asnumpy()
print B.shape

MM = mx.sym.sum(Bin, axis=1)

mm_exec = MM.bind(ctx=ctx_haha, args={'Bin' : B})

mm_exec.forward()
result = mm_exec.outputs[0].asnumpy()
print result.shape
print result


###################################### batch_dot ######################################

import mxnet as mx
ctx_haha = mx.gpu(0)

probs3dim = mx.symbol.Variable('probs3dim')
Bin = mx.symbol.Variable('Bin')

P = mx.nd.array([[[1],[2],[3],[4],[5]],[[2],[4],[6],[8],[10]]],ctx=ctx_haha) #2*5*1
P.asnumpy()
B = mx.nd.array([[[1,2,3,4,5],[1,3,5,7,9],[1,3,5,7,9]],[[1,2,3,4,5],[1,3,5,7,9],[1,3,5,7,9]]],ctx=ctx_haha) #2*3*5
B.asnumpy()


MM = mx.sym.batch_dot(Bin, probs3dim)

mm_exec = MM.bind(ctx=ctx_haha, args={'Bin' : B, 'probs3dim': P})

mm_exec.forward()
result = mm_exec.outputs[0].asnumpy()
print result

###################################### SwapAxis ######################################

import mxnet as mx
ctx_haha = mx.cpu()

Bin = mx.symbol.Variable('Bin')

B = mx.nd.array([[[1,2,3],[1,3,5],[1,2,5],[10,20,50],[1,2,5]],[[10,20,30],[1,3,5],[1,2,5],[1,2,5],[1,2,5]]],ctx=ctx_haha)
B.asnumpy()


MM = mx.sym.SwapAxis(Bin, dim1=1,dim2=2)

mm_exec = MM.bind(ctx=ctx_haha, args={'Bin' : B})

mm_exec.forward()
result = mm_exec.outputs[0].asnumpy()
print result


###################################### - ######################################

import mxnet as mx
ctx_haha = mx.gpu(0)

weight = mx.symbol.Variable('weight')

k = mx.nd.array([[1,2,3],[1,3,5]],ctx=ctx_haha)
k.asnumpy()

MM = 1-weight

mm_exec = MM.bind(ctx=ctx_haha, args={'weight' : k})

mm_exec.forward()
result = mm_exec.outputs[0].asnumpy()
print result

###################################### broadcast_mul & sum ######################################

import mxnet as mx
ctx_haha = mx.gpu(0)

M = mx.symbol.Variable('M')
k = mx.symbol.Variable('k')
w = mx.symbol.Variable('w')
memory = mx.nd.array([[[1,2,3],[1,3,5],[1,2,5],[1,2,5],[1,2,5]],[[1,2,3],[1,3,5],[1,2,5],[1,2,5],[1,2,5]]],ctx=ctx_haha)
memory.asnumpy()
weight = mx.nd.array([[1,2,3,4,5],[2,4,6,8,10]],ctx=ctx_haha)
weight.asnumpy()
key = mx.nd.array([[1,1,1],[2,2,2]],ctx=ctx_haha)
key.asnumpy()

M = mx.sym.broadcast_mul(mx.sym.expand_dims(w, axis=2), mx.sym.expand_dims(k, axis=1) )

mm_exec = M.bind(ctx=ctx_haha, args={'w' : weight, 'k': key})

mm_exec.forward()
result = mm_exec.outputs[0].asnumpy()
print result

###################################### SoftmaxActivation ######################################

import mxnet as mx
ctx_haha = mx.gpu(0)

Aout = mx.symbol.Variable('Aout')

a = mx.nd.array([[ 14.,  22.,  20.],[ 28.,  44.,  40.]],ctx=ctx_haha)
a.asnumpy()

softmax = mx.sym.SoftmaxActivation(Aout)

s_exec = softmax.bind(ctx=ctx_haha, args={'Aout' : a})

s_exec.forward()
result = s_exec.outputs[0].asnumpy()
print result

###################################### SoftmaxOutput ######################################

import mxnet as mx
ctx_haha = mx.gpu(0)

Aout = mx.symbol.Variable('Aout')
label = mx.symbol.Variable('label')

a = mx.nd.array([[ 14.,  22.,  20.],[ 28.,  44.,  40.]],ctx=ctx_haha)
a.asnumpy()
p = mx.nd.array([1,2],ctx=ctx_haha)
p.asnumpy()

softmax = mx.sym.SoftmaxOutput(data = Aout,label = label)

s_exec = softmax.bind(ctx=ctx_haha, args={'Aout' : a, 'label': p})

s_exec.forward()
result = s_exec.outputs[0].asnumpy()
print result

###################################### slice_axis ######################################
import mxnet as mx
ctx_haha = mx.gpu(0)

input = mx.symbol.Variable('input')

D = mx.nd.array([[1,2,3],[1,3,5]],ctx=ctx_haha)
D.asnumpy()

F = mx.symbol.slice_axis(input, axis=1, begin=2, end=3)


F_exec = F.bind(ctx=ctx_haha, args={'input': D})

F_exec.forward()
result = F_exec.outputs[0].asnumpy()
print result


###################################### Concat ######################################
import mxnet as mx
ctx_haha = mx.gpu(0)

left = mx.symbol.Variable('left')
right = mx.symbol.Variable('right')

l = mx.nd.array([[1,2,3],[1,3,5]], ctx=ctx_haha)
l.asnumpy()
r = mx.nd.array([[1,2,3],[1,3,5]], ctx=ctx_haha)
r.asnumpy()

cat = mx.sym.Concat(left, right, dim=1)

c_exec = cat.bind(ctx=ctx_haha, args={'left': l, 'right':r})

c_exec.forward()
result = c_exec.outputs[0].asnumpy()
print result


###################################### Fill ######################################
import mxnet as mx
import numpy as np

x = mx.nd.zeros((2, 3), dtype=np.float32)
x.asnumpy()
x[:] = 1.0
x.asnumpy()


###################################### Compute Cross Entropy using numpy ######################################

import numpy as np

a = np.array([[1,1],[0,1]])
b = np.array([[0.9,0.1],[0.5,0.5]])

-a*np.log(b)
np.sum(-a*np.log(b),axis= 0)




###################################### broadcast_mul & sum ######################################

import mxnet as mx
ctx_haha = mx.gpu(0)

probs3dim = mx.symbol.Variable('probs3dim')
Bin = mx.symbol.Variable('Bin')

P = mx.nd.array([[[1],[2],[3],[4],[5]],[[2],[4],[6],[8],[10]]],ctx=ctx_haha)
P.asnumpy()
B = mx.nd.array([[[1,2,3],[1,3,5],[1,2,5],[1,2,5],[1,2,5]],[[1,2,3],[1,3,5],[1,2,5],[1,2,5],[1,2,5]]],ctx=ctx_haha)
B.asnumpy()


MM = mx.sym.broadcast_mul(probs3dim, Bin)
MM = mx.sym.sum(MM, axis=1)

mm_exec = MM.bind(ctx=ctx_haha, args={'Bin' : B, 'probs3dim': P})

mm_exec.forward()
result = mm_exec.outputs[0].asnumpy()
print result

###################################### concat ######################################

import mxnet as mx

data = mx.nd.array([[ 0., 1., 2.], [ 3., 4., 5.]])
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
for dim in range(2):
    cat = mx.sym.Concat(a, b, dim=dim)
    exe = cat.bind(ctx=mx.cpu(), args={'a': data, 'b': data})
    exe.forward()
    out = exe.outputs[0]
    print "concat at dim = %d" % dim
    print "shape = %s" % (out.shape,)
    print "results = %s" % (out.asnumpy(),)


###################################### concat ######################################

import mxnet as mx

data = mx.nd.array([[ 0., 1., 2.], [ 3., 4., 5.]])
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
for dim in range(2):
    cat = mx.sym.Concat(a, b, dim=dim)
    exe = cat.bind(ctx=mx.cpu(), args={'a': data, 'b': data})
    exe.forward()
    out = exe.outputs[0]
    print "concat at dim = %d" % dim
    print "shape = %s" % (out.shape,)
    print "results = %s" % (out.asnumpy(),)

###################################### norm ######################################
