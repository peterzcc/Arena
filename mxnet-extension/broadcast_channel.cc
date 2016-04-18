/*!
* Copyright (c) 2016 by Contributors
* \file memory_update.cc
* \brief
* \author Xingjian Shi
*/

#include "./broadcast_channel-inl.h"

namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<cpu>(BroadcastChannelParam param) {
      return new BroadcastChannelOp<cpu>(param);
    }

    Operator* BroadcastChannelProp::CreateOperator(Context ctx) const {
      DO_BIND_DISPATCH(CreateOp, param_);
    }

    DMLC_REGISTER_PARAMETER(BroadcastChannelParam);

    MXNET_REGISTER_OP_PROPERTY(BroadcastChannel, BroadcastChannelProp)
      .add_argument("data", "Symbol", "Data to broadcast.")
      .add_arguments(BroadcastChannelParam::__FIELDS__())
      .describe("Broadcasting the data in the specific dimension");
  }  // namespace op
}  // namespace mxnet
