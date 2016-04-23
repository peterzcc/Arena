/*!
* Copyright (c) 2016 by Contributors
* \file complex_exchange.cu
* \brief
* \author Xingjian Shi
*/

#include "./complex_exchange-inl.h"


namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<gpu>(ComplexExchangeParam param) {
      return new ComplexExchangeOp<gpu>(param);
    }
  }  // namespace op
}  // namespace mxnet

