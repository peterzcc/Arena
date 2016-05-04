/*!
* Copyright (c) 2016 by Contributors
* \file fft-gpu-inl.h
* \brief
* \author Xingjian Shi
*/

#ifndef MXNET_OPERATOR_FFT2D_GPU_INL_H_
#define MXNET_OPERATOR_FFT2D_GPU_INL_H_

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
    namespace fft2d {
      enum FFTOpInputs { kData };
      enum FFTOpOutputs { kOut };
    }  // fft2d

    struct FFT2DParam : public dmlc::Parameter<FFT2DParam> {
      uint32_t batchsize;
      DMLC_DECLARE_PARAMETER(FFT2DParam) {
        DMLC_DECLARE_FIELD(batchsize).set_default(16).set_range(1, 256)
        .describe("Batchsize of the cuda operator.");
      }
    };

    /**
    * \brief This is the implementation of fft2d operator.
    * \tparam xpu The device that the op will be executed on.
    */
    template<typename xpu>
    class FFT2DOp : public Operator {
    public:
      explicit FFT2DOp(FFT2DParam p) {
        this->init_cufft_ = false;
        this->param_ = p;
      }

      ~FFT2DOp() {
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
        CHECK_EQ(req[fft2d::kOut], kWriteTo);
        CHECK_EQ(in_data.size(), 1);
        CHECK_EQ(out_data.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4> data = in_data[fft2d::kData].get<xpu, 4, real_t>(s);
        Tensor<xpu, 4> out = out_data[fft2d::kOut].get<xpu, 4, real_t>(s);
        CHECK_EQ(data.CheckContiguous(), true);
        CHECK_EQ(out.CheckContiguous(), true);
        if (!init_cufft_) {
          Init(s, in_data, out_data);
        }
        CHECK(0 == (data.shape_[0] * data.shape_[1]) % param_.batchsize);
        for (index_t i = 0; i < data.shape_[0] * data.shape_[1]; i += param_.batchsize) {
          CHECK_EQ(cufftExecR2C(plan, (cufftReal*)(data.dptr_ + i * data.shape_[2] * data.shape_[3]), 
            (cufftComplex*)(out.dptr_ + i * out.shape_[2] * out.shape_[3])), CUFFT_SUCCESS);
        }
        
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
        Tensor<xpu, 4> data = in_data[fft2d::kData].get<xpu, 4, real_t>(s);
        Tensor<xpu, 4> out = out_data[fft2d::kOut].get<xpu, 4, real_t>(s);
        int n[2] = { data.shape_[2], data.shape_[3] };
        // TODO This part may be memory-consuming
        if (data.shape_[0] * data.shape_[1] < param_.batchsize) {
          param_.batchsize = data.shape_[0] * data.shape_[1];
        }
        CHECK_EQ(cufftPlanMany(&plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, param_.batchsize), CUFFT_SUCCESS);
        init_cufft_ = true;
      }
      FFT2DParam param_;
      bool init_cufft_;
      cufftHandle plan;
    };  // class FFTOp

    // Decalre Factory function, used for dispatch specialization
    template<typename xpu>
    Operator* CreateOp(FFT2DParam type);

#if DMLC_USE_CXX11
    class FFT2DProp : public OperatorProperty {
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
        const TShape &dshape = in_shape->at(fft2d::kData);
        if (dshape.ndim() == 0) return false;
        CHECK(0 == (dshape[0] * dshape[1]) % param_.batchsize || dshape[0] * dshape[1] < param_.batchsize) << "In FFT2D, the dim[0] * dim[1] must be smaller than or be divide by batchsize. dim[0] = " <<
          dshape[0] << ", dim[1] = " << dshape[1] << ", batchsize = " << param_.batchsize;
        TShape oshape = dshape;
        oshape[3] = (dshape[3] / 2 + 1) * 2;
        out_shape->clear();
        out_shape->push_back(oshape);
        return true;
      }

      OperatorProperty* Copy() const override {
        auto ptr = new FFT2DProp();
        ptr->param_ = param_;
        return ptr;
      }

      std::string TypeString() const override {
        return "FFT2D";
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
      FFT2DParam param_;
    };
#endif  // DMLC_USE_CXX11
  }  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_FFT_GPU_INL_H_