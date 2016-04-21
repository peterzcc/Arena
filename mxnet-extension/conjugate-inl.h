/*!
* Copyright (c) 2016 by Contributors
* \file conjugate-inl.h
* \brief
* \author Xingjian Shi
*/

#ifndef MXNET_OPERATOR_CONJUGATE_INL_H_
#define MXNET_OPERATOR_CONJUGATE_INL_H_

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
    namespace conjugate {
      enum ConjugateInputs { kData };
      enum ConjugateOutputs { kOut };
    }  // conjugate

    struct ConjugateParam : public dmlc::Parameter<ConjugateParam> {
      DMLC_DECLARE_PARAMETER(ConjugateParam) {
      }
    };

    /**
    * \brief This is the implementation of conjugate operator.
    * \tparam xpu The device that the op will be executed on.
    */
    template<typename xpu>
    class ConjugateOp : public Operator {
    public:
      explicit ConjugateOp(ConjugateParam p) {
        this->param_ = p;
      }

      virtual void Forward(const OpContext &ctx,
        const std::vector<TBlob> &in_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &out_data,
        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(req[conjugate::kOut], kWriteTo);
        CHECK_EQ(in_data.size(), 1);
        CHECK_EQ(out_data.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4> data = in_data[conjugate::kData].get<xpu, 4, real_t>(s);
        Tensor<xpu, 4> out = out_data[conjugate::kOut].get<xpu, 4, real_t>(s);
        Assign(out, req[conjugate::kOut], complex_conjugate(data))
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
      ConjugateParam param_;
    };  // class FFTOp

    // Decalre Factory function, used for dispatch specialization
    template<typename xpu>
    Operator* CreateOp(ConjugateParam type);

#if DMLC_USE_CXX11
    class ConjugateProp : public OperatorProperty {
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
        const TShape &dshape = in_shape->at(conjugate::kData);
        if (dshape.ndim() == 0) return false;
        TShape oshape = dshape;
        out_shape->clear();
        out_shape->push_back(oshape);
        return true;
      }

      OperatorProperty* Copy() const override {
        auto ptr = new ConjugateProp();
        ptr->param_ = param_;
        return ptr;
      }

      std::string TypeString() const override {
        return "Conjugate";
      }

      std::vector<std::pair<int, void*> > ForwardInplaceOption(
        const std::vector<int> &in_data,
        const std::vector<void*> &out_data) const override {
        return{ { in_data[conjugate::kData], out_data[conjugate::kOut] } };
      }

      // decalre dependency and inplace optimization options
      std::vector<int> DeclareBackwardDependency(
        const std::vector<int> &out_grad,
        const std::vector<int> &in_data,
        const std::vector<int> &out_data) const override {
        return{ out_grad[conjugate::kOut], out_data[conjugate::kOut] };
      }

      Operator* CreateOperator(Context ctx) const override;

    private:
      ConjugateParam param_;
    };
#endif  // DMLC_USE_CXX11
  }  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_CONJUGATE_INL_H_