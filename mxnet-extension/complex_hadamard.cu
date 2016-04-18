/*!
* Copyright (c) 2016 by Contributors
* \file fft2d.cu
* \brief
* \author Xingjian Shi
*/

#include "./complex_hadamard-inl.h"


namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<gpu>(ComplexHadamardParam param) {
      return new ComplexHadamardOp<gpu>(param);
    }
  }  // namespace op
}  // namespace mxnet
