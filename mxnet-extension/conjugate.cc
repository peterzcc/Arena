/*!
* Copyright (c) 2016 by Contributors
* \file fft.cc
* \brief
* \author Xingjian Shi
*/

#include "./conjugate-inl.h"

namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<cpu>(ConjugateParam param) {
      return new ConjugateOp<cpu>(param);
    }

    Operator* ConjugateProp::CreateOperator(Context ctx) const {
      DO_BIND_DISPATCH(CreateOp, param_);
    }

    DMLC_REGISTER_PARAMETER(ConjugateParam);

    MXNET_REGISTER_OP_PROPERTY(Conjugate, ConjugateProp)
      .add_argument("data", "Symbol", "Input data to the conjugate operator.")
      .add_arguments(ConjugateParam::__FIELDS__())
      .describe("Ouput the conjugate of the inputs.");

  }  // namespace op
}  // namespace mxnet
