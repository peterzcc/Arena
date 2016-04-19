/*!
* Copyright (c) 2016 by Contributors
* \file broadcast_channel-inl.h
* \brief
* \author Xingjian Shi
*/

#ifndef MXNET_OPERATOR_BROADCAST_CHANNEL_INL_H_
#define MXNET_OPERATOR_BROADCAST_CHANNEL_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include "./operator_common.h"
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>

#define Broadcast_Samedim(X, S, dim) (reshape(broadcast_with_axis(X, dim, S[dim]), S))

namespace mxnet {
  namespace op {
    // Declare enumeration of input order to make code more intuitive.
    // // These enums are only visible within this header
    namespace broadcast_channel {
      enum BroadcastChannelOpInputs { kData };
      enum BroadcastChannelOpOutputs { kOut };
    }  // BroadcastChannel

    struct BroadcastChannelParam : public dmlc::Parameter<BroadcastChannelParam> {
      int dim;
      int size;
      DMLC_DECLARE_PARAMETER(BroadcastChannelParam) {
        DMLC_DECLARE_FIELD(dim).set_range(0, 4).set_default(1)
          .describe("the dimension to be broadcasted.");
        DMLC_DECLARE_FIELD(size).set_lower_bound(1).set_default(1)
          .describe("the size of the dimension after broadcasting.");
      }
    };

    /**
    * \brief This is the implementation of BroadcastChannel operator.
    * \tparam xpu The device that the op will be executed on.
    */
    template<typename xpu>
    class BroadcastChannelOp : public Operator {
    public:
      explicit BroadcastChannelOp(BroadcastChannelParam p) {
        this->param_ = p;
      }

      virtual void Forward(const OpContext &ctx,
        const std::vector<TBlob> &in_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &out_data,
        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(in_data.size(), 1);
        CHECK_EQ(out_data.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4> data = in_data[broadcast_channel::kData].get<xpu, 4, real_t>(s);
        Tensor<xpu, 4> out = out_data[broadcast_channel::kOut].get<xpu, 4, real_t>(s);
        Shape<4> new_shape = data.shape_;
        new_shape[param_.dim] = param_.size;
        Assign(out, req[broadcast_channel::kOut], Broadcast_Samedim(data, new_shape, param_.dim));
      }

      virtual void Backward(const OpContext &ctx,
        const std::vector<TBlob> &out_grad,
        const std::vector<TBlob> &in_data,
        const std::vector<TBlob> &out_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &in_grad,
        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
      }

    private:
      BroadcastChannelParam param_;
    };  // class BroadcastChannelOp

    // Decalre Factory function, used for dispatch specialization
    template<typename xpu>
    Operator* CreateOp(BroadcastChannelParam type);

#if DMLC_USE_CXX11
    class BroadcastChannelProp : public OperatorProperty {
    public:
      void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
        param_.Init(kwargs);
      }

      std::map<std::string, std::string> GetParams() const override {
        return param_.__DICT__();
      }

      bool InferShape(std::vector<TShape> *in_shape,
        std::vector<TShape> *out_shape,
        std::vector<TShape> *aux_shape) const override {
        using namespace mshadow;
        CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
        const TShape &dshape = in_shape->at(broadcast_channel::kData);
        if (dshape.ndim() == 0) return false;
        TShape oshape = dshape;
        oshape[param_.dim] = param_.size;
        out_shape->clear();
        out_shape->push_back(oshape);
        return true;
      }

      OperatorProperty* Copy() const override {
        auto ptr = new BroadcastChannelProp();
        ptr->param_ = param_;
        return ptr;
      }

      std::string TypeString() const override {
        return "BroadcastChannel";
      }

      // decalre dependency and inplace optimization options
      std::vector<int> DeclareBackwardDependency(
        const std::vector<int> &out_grad,
        const std::vector<int> &in_data,
        const std::vector<int> &out_data) const override {
        return{ out_grad[broadcast_channel::kOut], out_data[broadcast_channel::kOut] };
      }

      Operator* CreateOperator(Context ctx) const override;

    private:
      BroadcastChannelParam param_;
    };
#endif  // DMLC_USE_CXX11
  }  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_BROADCAST_CHANNEL_INL_H_