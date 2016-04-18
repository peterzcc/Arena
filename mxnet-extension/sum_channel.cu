/*!
* Copyright (c) 2016 by Contributors
* \file sum_channel.cu
* \brief
* \author Xingjian Shi
*/


#include "./sum_channel-inl.h"


namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<gpu>(SumChannelParam param) {
      return new SumChannelOp<gpu>(param);
    }
  }  // namespace op
}  // namespace mxnet

