/*!
* Copyright (c) 2016 by Contributors
* \file memory_update.cu
* \brief
* \author Xingjian Shi
*/
#include "./memory_update-inl.h"


namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<gpu>(MemoryUpdateParam param) {
      return new MemoryUpdateOp<gpu>(param);
    }
  }  // namespace op
}  // namespace mxnet

