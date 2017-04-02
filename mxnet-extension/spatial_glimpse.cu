/*!
* Copyright (c) 2016 by Contributors
* \file spatial_glimpse.cu
* \brief
* \author Xingjian Shi
*/

#include "./spatial_glimpse-inl.h"

namespace mxnet {
  namespace op {
    template<>
    Operator *CreateOp<gpu>(SpatialGlimpseParam param) {
      return new SpatialGlimpseOp<gpu>(param);
    }

  }  // namespace op
}  // namespace mxnet

