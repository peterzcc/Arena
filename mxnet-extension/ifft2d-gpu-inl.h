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

#define IFFT_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
      } while (0)

#define kIFFTMaxThreadsPerBlock 1024
#define kIFFTMaxGridNum 65535

#if defined(__CUDACC__)
/* TODO Use cufftXtSetCallback() */
template<typename Dtype>
__global__ void RescaleIRFFTInGradKernel(const int count, Dtype* in_grad, const int width, const bool is_odd) {
  int end = width;
  if (!is_odd) {
    end -= 2;
  }
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
    index < count;
    index += blockDim.x * gridDim.x) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    if (w >= 2 && w < end){
      in_grad[index] *= 2.0f;
    }
  }
}
#endif

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
      uint32_t batchsize;
      DMLC_DECLARE_PARAMETER(IFFT2DParam) {
        DMLC_DECLARE_FIELD(batchsize).set_default(32).set_range(1, 256)
          .describe("Batchsize of the cuda operator.");
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
        this->init_forward_cufft_ = false;
        this->init_backward_cufft_ = false;
        this->param_ = p;
      }

      ~IFFT2DOp() {
        if (init_forward_cufft_) {
          CHECK_EQ(cufftDestroy(forward_plan), CUFFT_SUCCESS);
        }
        if (init_backward_cufft_) {
          CHECK_EQ(cufftDestroy(backward_plan), CUFFT_SUCCESS);
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
        if (!init_forward_cufft_) {
          Init(out.shape_[0], out.shape_[1], out.shape_[2], out.shape_[3], 0);
        }
        CHECK(0 == (data.shape_[0] * data.shape_[1]) % param_.batchsize);
        for (int i = 0; i < data.shape_[0] * data.shape_[1]; i += param_.batchsize) {
          CHECK_EQ(cufftExecC2R(forward_plan, (cufftComplex*)(data.dptr_ + i * data.shape_[2] * data.shape_[3]), (cufftReal*)(out.dptr_ + i * out.shape_[2] * out.shape_[3])), CUFFT_SUCCESS);
        }
        if (init_forward_cufft_) {
          CHECK_EQ(cufftDestroy(forward_plan), CUFFT_SUCCESS);
          init_forward_cufft_ = false;
        }
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
        CHECK_NE(req[ifft2d::kData], kAddTo) << "AddTo not yet suported";
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4> igrad = in_grad[ifft2d::kData].get<xpu, 4, real_t>(s);
        Tensor<xpu, 4> ograd = out_grad[ifft2d::kOut].get<xpu, 4, real_t>(s);
        CHECK_EQ(igrad.CheckContiguous(), true);
        CHECK_EQ(ograd.CheckContiguous(), true);
        if (!init_backward_cufft_) {
          Init(ograd.shape_[0], ograd.shape_[1], ograd.shape_[2], ograd.shape_[3], 1);
        }
        CHECK(0 == (igrad.shape_[0] * igrad.shape_[1]) % param_.batchsize);
        for (int i = 0; i < igrad.shape_[0] * igrad.shape_[1]; i += param_.batchsize) {
          CHECK_EQ(cufftExecR2C(backward_plan, (cufftReal*)(ograd.dptr_ + i * ograd.shape_[2] * ograd.shape_[3]),
            (cufftComplex*)(igrad.dptr_ + i * igrad.shape_[2] * igrad.shape_[3])), CUFFT_SUCCESS);
        }
        if (init_backward_cufft_) {
          CHECK_EQ(cufftDestroy(backward_plan), CUFFT_SUCCESS);
          init_backward_cufft_ = false;
        }
#if defined(__CUDACC__)
        const int count = igrad.shape_.Size();
        const int gridSize = (count + kIFFTMaxThreadsPerBlock - 1) / kIFFTMaxThreadsPerBlock;
        dim3 dimGrid(kIFFTMaxGridNum, (gridSize + kIFFTMaxGridNum - 1) / kIFFTMaxGridNum);
        dim3 dimBlock(kIFFTMaxThreadsPerBlock);
        cudaStream_t stream = Stream<gpu>::GetStream(igrad.stream_);
        RescaleIRFFTInGradKernel<real_t> << <dimGrid, dimBlock, 0, stream >> >(count, igrad.dptr_, igrad.shape_[3], ograd.shape_[3]%2);
        IFFT_CUDA_CHECK(cudaPeekAtLastError());
#endif
        igrad /= static_cast<real_t>(ograd.shape_[2] * ograd.shape_[3]);
      }

    private:
      inline void Init(int num, int channel, int rows, int cols, int typ) {
        int n[2] = { rows, cols };
        if (num * channel < param_.batchsize) {
          param_.batchsize = num * channel;
        }
        if (0 == typ){
          CHECK_EQ(cufftPlanMany(&forward_plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, param_.batchsize), CUFFT_SUCCESS);
          init_forward_cufft_ = true;
        }
        else {
          CHECK_EQ(cufftPlanMany(&backward_plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, param_.batchsize), CUFFT_SUCCESS);
          init_backward_cufft_ = true;
        }
        
      }
      IFFT2DParam param_;
      bool init_forward_cufft_;
      bool init_backward_cufft_;
      cufftHandle forward_plan;
      cufftHandle backward_plan;
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
        CHECK(0 == (dshape[0] * dshape[1]) % param_.batchsize || dshape[0] * dshape[1] < param_.batchsize) << "In FFT2D, the dim[0] * dim[1] must be smaller than or be divide by batchsize. dim[0] = " <<
          dshape[0] << ", dim[1] = " << dshape[1] << ", batchsize = " << param_.batchsize;
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
        return{ out_grad[ifft2d::kOut] };
      }

      Operator* CreateOperator(Context ctx) const override;

    private:
      IFFT2DParam param_;
    };
#endif  // DMLC_USE_CXX11
  }  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_FFT_GPU_INL_H_