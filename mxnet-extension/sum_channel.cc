/*!
* Copyright (c) 2016 by Contributors
* \file fft.cc
* \brief
* \author Xingjian Shi
*/

#include "./sum_channel-inl.h"

namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<cpu>(SumChannelParam param) {
      return new SumChannelOp<cpu>(param);
    }

    Operator* SumChannelProp::CreateOperator(Context ctx) const {
      DO_BIND_DISPATCH(CreateOp, param_);
    }

    DMLC_REGISTER_PARAMETER(SumChannelParam);

    MXNET_REGISTER_OP_PROPERTY(SumChannel, SumChannelProp)
      .add_argument("data", "Symbol", "Input data.")
      .add_arguments(SumChannelParam::__FIELDS__())
      .describe("Perform summation over a specific channel in the input.");

  }  // namespace op
}  // namespace mxnet
