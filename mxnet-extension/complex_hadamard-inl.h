/*!
* Copyright (c) 2016 by Contributors
* \file complex_hadamard-inl.h
* \brief
* \author Xingjian Shi
*/

#ifndef MXNET_OPERATOR_COMPLEX_HADAMARD_INL_H_
#define MXNET_OPERATOR_COMPLEX_HADAMARD_INL_H_

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
    namespace complex_hadamard {
      enum ComplexHadamardOpInputs { kLData, kRData };
      enum ComplexHadamardOpOutputs { kOut };
    }  // ComplexHadamard

    struct ComplexHadamardParam : public dmlc::Parameter<ComplexHadamardParam> {
      DMLC_DECLARE_PARAMETER(ComplexHadamardParam) {
      }
    };

    /**
    * \brief This is the implementation of ComplexHadamard operator.
    * \tparam xpu The device that the op will be executed on.
    */
    template<typename xpu>
    class ComplexHadamardOp : public Operator {
    public:
      explicit ComplexHadamardOp(ComplexHadamardParam p) {
        this->param_ = p;
      }

      virtual void Forward(const OpContext &ctx,
        const std::vector<TBlob> &in_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &out_data,
        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(in_data.size(), 2);
        CHECK_EQ(out_data.size(), 1);
        Stream<gpu> *s = ctx.get_stream<gpu>();
        Tensor<gpu, 4> ldata = in_data[complex_hadamard::kLData].get<gpu, 4, real_t>(s);
        Tensor<gpu, 4> rdata = in_data[complex_hadamard::kRData].get<gpu, 4, real_t>(s);
        Tensor<gpu, 4> out = out_data[complex_hadamard::kOut].get<gpu, 4, real_t>(s);
        Assign(out, req[complex_hadamard::kOut], complex_hadamard_product(ldata, rdata));
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
      ComplexHadamardParam param_;
    };  // class ComplexHadamardOp

    // Decalre Factory function, used for dispatch specialization
    template<typename xpu>
    Operator* CreateOp(ComplexHadamardParam type);

#if DMLC_USE_CXX11
    class ComplexHadamardProp : public OperatorProperty {
    public:
      void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
        param_.Init(kwargs);
      }

      std::vector<std::string> ListArguments() const override {
        return{ "ldata", "rdata" };
      }

      std::map<std::string, std::string> GetParams() const override {
        return param_.__DICT__();
      }

      bool InferShape(std::vector<TShape> *in_shape,
        std::vector<TShape> *out_shape,
        std::vector<TShape> *aux_shape) const override {
        using namespace mshadow;
        CHECK_EQ(in_shape->size(), 2) << "Input:[ldata, rdata]";
        const TShape &ldshape = in_shape->at(complex_hadamard::kLData);
        const TShape &rdshape = in_shape->at(complex_hadamard::kRData);
        if (ldshape.ndim() == 0) return false;
        CHECK_EQ(ldshape, rdshape) << "Two inputs must have the same shape!";
        TShape oshape = ldshape;
        out_shape->clear();
        out_shape->push_back(oshape);
        return true;
      }

      OperatorProperty* Copy() const override {
        auto ptr = new ComplexHadamardProp();
        ptr->param_ = param_;
        return ptr;
      }

      std::string TypeString() const override {
        return "ComplexHadamard";
      }

      // decalre dependency and inplace optimization options
      std::vector<int> DeclareBackwardDependency(
        const std::vector<int> &out_grad,
        const std::vector<int> &in_data,
        const std::vector<int> &out_data) const override {
        return{ out_grad[complex_hadamard::kOut], out_data[complex_hadamard::kOut] };
      }

      Operator* CreateOperator(Context ctx) const override;

    private:
      ComplexHadamardParam param_;
    };
#endif  // DMLC_USE_CXX11
  }  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_FFT_GPU_INL_H_