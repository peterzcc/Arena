/*!
* Copyright (c) 2016 by Contributors
* \file memory_update-inl.h
* \brief
* \author Xingjian Shi
*/

#ifndef MXNET_OPERATOR_MEMORY_UPDATE_INL_H_
#define MXNET_OPERATOR_MEMORY_UPDATE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include "./operator_common.h"
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>

#define Broadcast_Scalar(X, S) (broadcast<0>(reshape(broadcast_with_axis(X, 0, S[0]), Shape1(S[0])), S))
#define Broadcast_Samedim(X, S, dim) (reshape(broadcast_with_axis(X, dim, S[dim]), S))



namespace mxnet {
  namespace op {
    // Declare enumeration of input order to make code more intuitive.
    // // These enums are only visible within this header
    namespace memory_update {
      enum MemoryUpdateOpInputs { kData, kUpdate, kFlag, kFactor };
      enum MemoryUpdateOpOutputs { kOut };
    }  // memory_update

    struct MemoryUpdateParam : public dmlc::Parameter<MemoryUpdateParam> {
      DMLC_DECLARE_PARAMETER(MemoryUpdateParam) {
      }
    };

    /**
    * \brief This is the implementation of memory update operator.
    * \tparam xpu The device that the op will be executed on.
    */
    template<typename xpu>
    class MemoryUpdateOp : public Operator {
    public:
      explicit MemoryUpdateOp(MemoryUpdateParam p) {
        this->param_ = p;
      }

      virtual void Forward(const OpContext &ctx,
        const std::vector<TBlob> &in_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &out_data,
        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(in_data.size(), 4);
        CHECK_EQ(out_data.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4, real_t> data = in_data[memory_update::kData].get<xpu, 4, real_t>(s);
        Tensor<xpu, 4, real_t> update = in_data[memory_update::kUpdate].get<xpu, 4, real_t>(s);
        Tensor<xpu, 1, real_t> flag = in_data[memory_update::kFlag].get<xpu, 1, real_t>(s);
        Tensor<xpu, 1, real_t> factor = in_data[memory_update::kFactor].get<xpu, 1, real_t>(s);
        Tensor<xpu, 4, real_t> out = out_data[memory_update::kOut].get<xpu, 4, real_t>(s);
        Assign(out, req[memory_update::kData], select_among_three(data, (1 - Broadcast_Scalar(factor, data.shape_)) * data + 
          Broadcast_Scalar(factor, data.shape_) * Broadcast_Samedim(update, data.shape_, 0),
          Broadcast_Samedim(update, data.shape_, 0), flag));
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
      MemoryUpdateParam param_;
    };  // class MemoryUpdateOp

    // Decalre Factory function, used for dispatch specialization
    template<typename xpu>
    Operator* CreateOp(MemoryUpdateParam type);

#if DMLC_USE_CXX11
    class MemoryUpdateProp : public OperatorProperty {
    public:

      std::vector<std::string> ListArguments() const override {
        return{ "data", "update", "flag", "factor" };
      }

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
        CHECK_EQ(in_shape->size(), 4) << "Input:[data, update, flag, factor]";
        const TShape &dshape = in_shape->at(memory_update::kData);
        const TShape &update_shape = in_shape->at(memory_update::kUpdate);
        const TShape &flag_shape = in_shape->at(memory_update::kFlag);
        const TShape &factor_shape = in_shape->at(memory_update::kFactor);
        CHECK_EQ(update_shape[0], 1) << "The first dimension of update must be one!";
        CHECK_EQ(flag_shape[0], dshape[0]) << "Flag number must be equal to the number of memory elements.";
        CHECK_EQ(factor_shape[0], 1) << "Factor number must be equal to 1.";
        TShape oshape = dshape;
        out_shape->clear();
        out_shape->push_back(oshape);
        return true;
      }

      OperatorProperty* Copy() const override {
        auto ptr = new MemoryUpdateProp();
        ptr->param_ = param_;
        return ptr;
      }

      std::string TypeString() const override {
        return "MemoryUpdate";
      }

      std::vector<std::pair<int, void*> > ForwardInplaceOption(
        const std::vector<int> &in_data,
        const std::vector<void*> &out_data) const override {
        return{ { in_data[memory_update::kData], out_data[memory_update::kOut] } };
      }

      // decalre dependency and inplace optimization options
      std::vector<int> DeclareBackwardDependency(
        const std::vector<int> &out_grad,
        const std::vector<int> &in_data,
        const std::vector<int> &out_data) const override {
        return{ out_grad[memory_update::kOut], out_data[memory_update::kOut] };
      }

      Operator* CreateOperator(Context ctx) const override;

    private:
      MemoryUpdateParam param_;
    };
#endif  // DMLC_USE_CXX11
  }  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_MEMORY_UPDATE_INL_H_