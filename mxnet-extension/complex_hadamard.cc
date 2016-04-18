/*!
* Copyright (c) 2016 by Contributors
* \file fft.cc
* \brief
* \author Xingjian Shi
*/

#include "./complex_hadamard-inl.h"

namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<cpu>(ComplexHadamardParam param) {
      return new ComplexHadamardOp<cpu>(param);
    }

    Operator* ComplexHadamardProp::CreateOperator(Context ctx) const {
      DO_BIND_DISPATCH(CreateOp, param_);
    }

    DMLC_REGISTER_PARAMETER(ComplexHadamardParam);

    MXNET_REGISTER_OP_PROPERTY(ComplexHadamard, ComplexHadamardProp)
      .add_argument("ldata", "Symbol", "Left input data to the hadamard product operator.")
      .add_argument("rdata", "Symbol", "Right input data to the hadamard product operator.")
      .add_arguments(ComplexHadamardParam::__FIELDS__())
      .describe("Get the complex hadamard product of the two inputs.");

  }  // namespace op
}  // namespace mxnet
