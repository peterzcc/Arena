/*!
* Copyright (c) 2016 by Contributors
* \file fft-gpu-inl.h
* \brief
* \author Xingjian Shi
*/

#ifndef MXNET_OPERATOR_IFFT2D_GPU_INL_H_
#define MXNET_OPERATOR_IFFT2D_GPU_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include "./operator_common.h"
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <cufft.h>

namespace mxnet {
  namespace op {
    // Declare enumeration of input order to make code more intuitive.
    // // These enums are only visible within this header
    namespace ifft2d {
      enum IFFTOpInputs { kData };
      enum IFFTOpOutputs { kOut };
    }  // ifft2d

    struct IFFT2DParam : public dmlc::Parameter<IFFT2DParam> {
      TShape output_shape;
      DMLC_DECLARE_PARAMETER(IFFT2DParam) {
        DMLC_DECLARE_FIELD(output_shape)
          .set_expect_ndim(2).enforce_nonzero()
          .describe("Size of the spatial_output: (H, W)");
      }
    };

    /**
    * \brief This is the implementation of ifft2d operator.
    * \tparam xpu The device that the op will be executed on.
    */
    template<typename xpu>
    class IFFT2DOp : public Operator {
    public:
      explicit IFFT2DOp(IFFT2DParam p) {
        this->init_cufft_ = false;
        this->param_ = p;
      }

      ~IFFT2DOp() {
        if (init_cufft_) {
          CHECK_EQ(cufftDestroy(plan), CUFFT_SUCCESS);
        }
      }

      virtual void Forward(const OpContext &ctx,
        const std::vector<TBlob> &in_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &out_data,
        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(req[ifft2d::kOut], kWriteTo);
        CHECK_EQ(in_data.size(), 1);
        CHECK_EQ(out_data.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4> data = in_data[ifft2d::kData].get<xpu, 4, real_t>(s);
        Tensor<xpu, 4> out = out_data[ifft2d::kOut].get<xpu, 4, real_t>(s);
        CHECK_EQ(data.CheckContiguous(), true);
        CHECK_EQ(out.CheckContiguous(), true);
        if (!init_cufft_) {
          Init(s, in_data, out_data);
        }
        CHECK_EQ(cufftExecC2R(plan, (cufftComplex*)data.dptr_, (cufftReal*)out.dptr_), CUFFT_SUCCESS);
        out /= static_cast<real_t>(param_.output_shape[0] * param_.output_shape[1]);
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
        CHECK_EQ(out_grad.size(), 1);
        CHECK(in_data.size() == 1 && in_grad.size() == 1);
        CHECK_EQ(req.size(), 1);
        // LOG(FATAL) << "Backward not implemented yet!";
      }

    private:
      inline void Init(mshadow::Stream<xpu> *s,
        const std::vector<TBlob> &in_data,
        const std::vector<TBlob> &out_data) {
        using namespace mshadow;
        using namespace mshadow::expr;
        Tensor<xpu, 4> data = in_data[ifft2d::kData].get<xpu, 4, real_t>(s);
        Tensor<xpu, 4> out = out_data[ifft2d::kOut].get<xpu, 4, real_t>(s);
        int n[2] = { this->param_.output_shape[0], this->param_.output_shape[1] };
        // TODO This part may be memory-consuming
        CHECK_EQ(cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, data.shape_[0] * data.shape_[1]), CUFFT_SUCCESS);
        init_cufft_ = true;
      }
      IFFT2DParam param_;
      bool init_cufft_;
      cufftHandle plan;
    };  // class FFTOp

    // Decalre Factory function, used for dispatch specialization
    template<typename xpu>
    Operator* CreateOp(IFFT2DParam type);

#if DMLC_USE_CXX11
    class IFFT2DProp : public OperatorProperty {
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
        const TShape &dshape = in_shape->at(ifft2d::kData);
        if (dshape.ndim() == 0) return false;
        TShape oshape = dshape;
        oshape[2] = param_.output_shape[0];
        oshape[3] = param_.output_shape[1];
        out_shape->clear();
        out_shape->push_back(oshape);
        return true;
      }

      OperatorProperty* Copy() const override {
        auto ptr = new IFFT2DProp();
        ptr->param_ = param_;
        return ptr;
      }

      std::string TypeString() const override {
        return "IFFT2D";
      }

      // decalre dependency and inplace optimization options
      std::vector<int> DeclareBackwardDependency(
        const std::vector<int> &out_grad,
        const std::vector<int> &in_data,
        const std::vector<int> &out_data) const override {
        return{ out_data[ifft2d::kOut] };
      }

      Operator* CreateOperator(Context ctx) const override;

    private:
      IFFT2DParam param_;
    };
#endif  // DMLC_USE_CXX11
  }  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_FFT_GPU_INL_H_