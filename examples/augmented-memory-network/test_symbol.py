###################################### broadcast_mul & sum ######################################

import mxnet as mx

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
