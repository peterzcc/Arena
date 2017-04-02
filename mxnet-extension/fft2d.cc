/*!
* Copyright (c) 2016 by Contributors
* \file fft.cc
* \brief
* \author Xingjian Shi
*/

#include "./fft2d-gpu-inl.h"

namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<cpu>(FFT2DParam param) {
      LOG(FATAL) << "FFT2D is currently only supported "
        "on GPU with cuFFT.";
      return new FFT2DOp<cpu>(param);
    }

    Operator* FFT2DProp::CreateOperator(Context ctx) const {
      DO_BIND_DISPATCH(CreateOp, param_);
    }

    DMLC_REGISTER_PARAMETER(FFT2DParam);

    MXNET_REGISTER_OP_PROPERTY(FFT2D, FFT2DProp)
      .add_argument("data", "Symbol", "Input data to the 2D FFT operator.")
      .add_arguments(FFT2DParam::__FIELDS__())
      .describe("Perform 2D R2C FFT on the inputs.");

  }  // namespace op
}  // namespace mxnet
