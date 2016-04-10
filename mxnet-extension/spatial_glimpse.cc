/*!
 * Copyright (c) 2016 by Contributors
 * \file spatial_glimpse.cc
 * \brief
 * \author Xingjian Shi
*/
#include "./spatial_glimpse-inl.h"

namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<cpu>(SpatialGlimpseParam param) {
      return new SpatialGlimpseOp<cpu>(param);
    }

    Operator* SpatialGlimpseProp::CreateOperator(Context ctx) const {
      DO_BIND_DISPATCH(CreateOp, param_);
    }
    
    DMLC_REGISTER_PARAMETER(SpatialGlimpseParam);

    MXNET_REGISTER_OP_PROPERTY(SpatialGlimpse, SpatialGlimpseProp)
      .describe("Perform spatial glimpse on the inputs.")
      .add_argument("data", "Symbol", "Input data to the spatial glimpse operator.")
      .add_argument("center", "Symbol", "Center of the spatial glimpse. Shape is (nroi, 2), each row contains (cx, cy).")
      .add_argument("size", "Symbol", "Initial Size of the spatial glimpse. Shape is (nroi, 2), each row contains (sx, sy).")
      .add_arguments(SpatialGlimpseParam::__FIELDS__());

  }  // namespace op
}  // namespace mxnet
