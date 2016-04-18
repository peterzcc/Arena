/*!
* Copyright (c) 2016 by Contributors
* \file broadcast_channel.cu
* \brief
* \author Xingjian Shi
*/

#include "./broadcast_channel-inl.h"


namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<gpu>(BroadcastChannelParam param) {
      return new BroadcastChannelOp<gpu>(param);
    }
  }  // namespace op
}  // namespace mxnet

