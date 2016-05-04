/*!
* Copyright (c) 2016 by Contributors
* \file memory_choose-inl.h
* \brief
* \author Xingjian Shi
*/

#ifndef MXNET_OPERATOR_MEMORY_CHOOSE_INL_H_
#define MXNET_OPERATOR_MEMORY_CHOOSE_INL_H_

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
    namespace memory_choose {
      enum MemoryChooseOpInputs { kData, kIndex };
      enum MemoryChooseOpOutputs { kOut };
    }  // memory_choose

    struct MemoryChooseParam : public dmlc::Parameter<MemoryChooseParam> {
      DMLC_DECLARE_PARAMETER(MemoryChooseParam) {
      }
    };

    /**
    * \brief This is the implementation of memory choose operator.
    * \tparam xpu The device that the op will be executed on.
    */
    template<typename xpu>
    class MemoryChooseOp : public Operator {
    public:
      explicit MemoryChooseOp(MemoryChooseParam p) {
        this->param_ = p;
      }

      virtual void Forward(const OpContext &ctx,
        const std::vector<TBlob> &in_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &out_data,
        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(req[memory_choose::kOut], kWriteTo);
        CHECK_EQ(in_data.size(), 2);
        CHECK_EQ(out_data.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4, real_t> data = in_data[memory_choose::kData].get<xpu, 4, real_t>(s);
        Tensor<xpu, 1, real_t> index = in_data[memory_choose::kIndex].get<xpu, 1, real_t>(s);
        Tensor<xpu, 4, real_t> out = out_data[memory_choose::kOut].get<xpu, 4, real_t>(s);
        Assign(out, req[memory_choose::kData], choose_tensor(data, index));
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
      MemoryChooseParam param_;
    };  // class MemoryChooseOp

    // Decalre Factory function, used for dispatch specialization
    template<typename xpu>
    Operator* CreateOp(MemoryChooseParam type);

#if DMLC_USE_CXX11
    class MemoryChooseProp : public OperatorProperty {
    public:
      
      std::vector<std::string> ListArguments() const override {
        return{ "data", "index" };
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
        CHECK_EQ(in_shape->size(), 2) << "Input:[data, index]";
        const TShape &dshape = in_shape->at(memory_choose::kData);
        const TShape &indshape = in_shape->at(memory_choose::kIndex);
        TShape oshape = dshape;
        oshape[0] = indshape[0];
        out_shape->clear();
        out_shape->push_back(oshape);
        return true;
      }

      OperatorProperty* Copy() const override {
        auto ptr = new MemoryChooseProp();
        ptr->param_ = param_;
        return ptr;
      }

      std::string TypeString() const override {
        return "MemoryChoose";
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
      MemoryChooseParam param_;
    };
#endif  // DMLC_USE_CXX11
  }  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_MEMORY_CHOOSE_INL_H_