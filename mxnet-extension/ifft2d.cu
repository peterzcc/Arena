/*!
* Copyright (c) 2016 by Contributors
* \file fft2d.cu
* \brief
* \author Xingjian Shi
*/
#include "./ifft2d-gpu-inl.h"


namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<gpu>(IFFT2DParam param) {
      return new IFFT2DOp<gpu>(param);
    }
  }  // namespace op
}  // namespace mxnet

