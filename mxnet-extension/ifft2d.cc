/*!
* Copyright (c) 2016 by Contributors
* \file fft.cc
* \brief
* \author Xingjian Shi
*/

#include "./ifft2d-gpu-inl.h"

namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<cpu>(IFFT2DParam param) {
      LOG(FATAL) << "IFFT2D is currently only supported "
        "on GPU with cuFFT.";
      return new IFFT2DOp<cpu>(param);
    }

    Operator* IFFT2DProp::CreateOperator(Context ctx) const {
      DO_BIND_DISPATCH(CreateOp, param_);
    }

    DMLC_REGISTER_PARAMETER(IFFT2DParam);

    MXNET_REGISTER_OP_PROPERTY(IFFT2D, IFFT2DProp)
      .add_argument("data", "Symbol", "Input data to the 2D Complex to Real IFFT operator.")
      .add_arguments(IFFT2DParam::__FIELDS__())
      .describe("Perform 2D C2R IFFT on the inputs.");

  }  // namespace op
}  // namespace mxnet
