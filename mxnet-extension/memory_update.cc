/*!
* Copyright (c) 2016 by Contributors
* \file memory_update.cc
* \brief
* \author Xingjian Shi
*/

#include "./memory_update-inl.h"

namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<cpu>(MemoryUpdateParam param) {
      return new MemoryUpdateOp<cpu>(param);
    }

    Operator* MemoryUpdateProp::CreateOperator(Context ctx) const {
      DO_BIND_DISPATCH(CreateOp, param_);
    }

    DMLC_REGISTER_PARAMETER(MemoryUpdateParam);

    MXNET_REGISTER_OP_PROPERTY(MemoryUpdate, MemoryUpdateProp)
      .add_argument("data", "Symbol", "Old Memory.")
      .add_argument("update", "Symbol", "The memory content to insert.")
      .add_argument("flag", "Symbol", "Updating flags.")
      .add_argument("factor", "Symbol", "Updating factors")
      .add_arguments(MemoryUpdateParam::__FIELDS__())
      .describe("Update the memory unit based on the input contents.");
  }  // namespace op
}  // namespace mxnet
