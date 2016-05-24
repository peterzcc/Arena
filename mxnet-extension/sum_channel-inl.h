/*!
* Copyright (c) 2016 by Contributors
* \file sum_channel-inl.h
* \brief
* \author Xingjian Shi
*/

#ifndef MXNET_OPERATOR_SUM_CHANNEL_INL_H_
#define MXNET_OPERATOR_SUM_CHANNEL_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include "./operator_common.h"
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>

namespace mxnet {
  namespace op {
    // Declare enumeration of input order to make code more intuitive.
    // // These enums are only visible within this header
    namespace sum_channel {
      enum SumChannelOpInputs { kData };
      enum SumChannelOpOutputs { kOut };
    }  // SumChannel

    struct SumChannelParam : public dmlc::Parameter<SumChannelParam> {
      int dim;
      DMLC_DECLARE_PARAMETER(SumChannelParam) {
        DMLC_DECLARE_FIELD(dim).set_range(0, 4).set_default(1)
          .describe("the dimension to take sum.");
      }
    };

    /**
    * \brief This is the implementation of sum channel operator.
    * \tparam xpu The device that the op will be executed on.
    */
    template<typename xpu>
    class SumChannelOp : public Operator {
    public:
      explicit SumChannelOp(SumChannelParam p) {
        this->param_ = p;
      }

      virtual void Forward(const OpContext &ctx,
        const std::vector<TBlob> &in_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &out_data,
        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(req[sum_channel::kOut], kWriteTo);
        CHECK_EQ(in_data.size(), 1);
        CHECK_EQ(out_data.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4> data = in_data[sum_channel::kData].get<xpu, 4, real_t>(s);
        Tensor<xpu, 4> out = out_data[sum_channel::kOut].get<xpu, 4, real_t>(s);
        Shape<5> broadcasting_shape;
        for (int i = 0; i < 5; i++) {
          if (i < param_.dim + 1){
            broadcasting_shape[i] = data.shape_[i];
          }
          else if (i > param_.dim + 1) {
            broadcasting_shape[i] = data.shape_[i - 1];
          }
          else{
            broadcasting_shape[i] = 1;
          }
        }
        Assign(out, req[sum_channel::kOut], (mshadow::expr::reduce_with_axis<mshadow::red::sum, false>(reshape(data, broadcasting_shape), param_.dim)));
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
        // LOG(FATAL) << "Backward not implemented yet!";
      }

    private:
      SumChannelParam param_;
    };  // class SumChannelOp

    // Decalre Factory function, used for dispatch specialization
    template<typename xpu>
    Operator* CreateOp(SumChannelParam type);

#if DMLC_USE_CXX11
    class SumChannelProp : public OperatorProperty {
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
        const TShape &dshape = in_shape->at(sum_channel::kData);
        if (dshape.ndim() == 0) return false;
        TShape oshape = dshape;
        oshape[param_.dim] = 1;
        out_shape->clear();
        out_shape->push_back(oshape);
        return true;
      }

      OperatorProperty* Copy() const override {
        auto ptr = new SumChannelProp();
        ptr->param_ = param_;
        return ptr;
      }

      std::string TypeString() const override {
        return "SumChannel";
      }

      // decalre dependency and inplace optimization options
      std::vector<int> DeclareBackwardDependency(
        const std::vector<int> &out_grad,
        const std::vector<int> &in_data,
        const std::vector<int> &out_data) const override {
        return{ };
      }

      Operator* CreateOperator(Context ctx) const override;

    private:
      SumChannelParam param_;
    };
#endif  // DMLC_USE_CXX11
  }  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_SUM_CHANNEL_INL_H_