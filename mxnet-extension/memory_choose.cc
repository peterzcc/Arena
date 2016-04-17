/*!
* Copyright (c) 2016 by Contributors
* \file fft.cc
* \brief
* \author Xingjian Shi
*/

#include "./memory_choose-inl.h"

namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<cpu>(MemoryChooseParam param) {
      return new MemoryChooseOp<cpu>(param);
    }

    Operator* MemoryChooseProp::CreateOperator(Context ctx) const {
      DO_BIND_DISPATCH(CreateOp, param_);
    }

    DMLC_REGISTER_PARAMETER(MemoryChooseParam);

    MXNET_REGISTER_OP_PROPERTY(MemoryChoose, MemoryChooseProp)
      .add_argument("data", "Symbol", "Memory to the operator.")
      .add_argument("index", "Symbol", "Index of the reading elements.")
      .add_arguments(MemoryChooseParam::__FIELDS__())
      .describe("Read a specific element of the memory unit.");
  }  // namespace op
}  // namespace mxnet
