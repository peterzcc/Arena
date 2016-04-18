/*!
* Copyright (c) 2016 by Contributors
* \file fft2d.cu
* \brief
* \author Xingjian Shi
*/

#include "./conjugate-inl.h"


namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<gpu>(ConjugateParam param) {
      return new ConjugateOp<gpu>(param);
    }
  }  // namespace op
}  // namespace mxnet

