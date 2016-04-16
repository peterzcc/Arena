/*!
* Copyright (c) 2016 by Contributors
* \file fft2d.cu
* \brief
* \author Xingjian Shi
*/
#include "./fft2d-gpu-inl.h"


namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<gpu>(FFT2DParam param) {
      return new FFT2DOp<gpu>(param);
    }
  }  // namespace op
}  // namespace mxnet

