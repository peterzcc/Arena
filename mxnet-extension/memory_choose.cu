/*!
* Copyright (c) 2016 by Contributors
* \file memory_choose.cu
* \brief
* \author Xingjian Shi
*/
#include "./memory_choose-inl.h"


namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<gpu>(MemoryChooseParam param) {
      return new MemoryChooseOp<gpu>(param);
    }
  }  // namespace op
}  // namespace mxnet

