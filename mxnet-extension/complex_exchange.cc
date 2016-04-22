/*!
* Copyright (c) 2016 by Contributors
* \file complex_exchange.cc
* \brief
* \author Xingjian Shi
*/

#include "./complex_exchange-inl.h"

namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<cpu>(ComplexExchangeParam param) {
      return new ComplexExchangeOp<cpu>(param);
    }

    Operator* ComplexExchangeProp::CreateOperator(Context ctx) const {
      DO_BIND_DISPATCH(CreateOp, param_);
    }

    DMLC_REGISTER_PARAMETER(ComplexExchangeParam);

    MXNET_REGISTER_OP_PROPERTY(ComplexExchange, ComplexExchangeProp)
      .add_argument("data", "Symbol", "Input data to the ComplexExchange operator.")
      .add_arguments(ComplexExchangeParam::__FIELDS__())
      .describe("Ouput the ComplexExchange of the inputs.");

  }  // namespace op
}  // namespace mxnet
