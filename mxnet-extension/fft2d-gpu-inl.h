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

#define FRCNN_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    } while (0)

#define FRCNN_DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define FRCNN_NUM_THREADS 1024

template<typename Dtype>
__global__ void RescaleRFFTOutGradKernel(const int count, Dtype* out_grad, const int width) {
  int end = width;
  if (width % 2 == 0) {
    end -= 2;
  }
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
    index < count;
    index += blockDim.x * gridDim.x) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    if (w >= 2 && w < end){
      out_grad[index] /= 2;
    }
  }
}

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
        this->init_forward_cufft_ = false;
        this->init_backward_cufft_ = false;
        this->param_ = p;
      }

      ~FFT2DOp() {
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
        CHECK_EQ(req[fft2d::kOut], kWriteTo);
        CHECK_EQ(in_data.size(), 1);
        CHECK_EQ(out_data.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4> data = in_data[fft2d::kData].get<xpu, 4, real_t>(s);
        Tensor<xpu, 4> out = out_data[fft2d::kOut].get<xpu, 4, real_t>(s);
        CHECK_EQ(data.CheckContiguous(), true);
        CHECK_EQ(out.CheckContiguous(), true);
        if (!init_forward_cufft_) {
          Init(data.shape_[0], data.shape_[1], data.shape_[2], data.shape_[3], 0);
        }
        CHECK(0 == (data.shape_[0] * data.shape_[1]) % param_.batchsize);
        for (index_t i = 0; i < data.shape_[0] * data.shape_[1]; i += param_.batchsize) {
          CHECK_EQ(cufftExecR2C(forward_plan, (cufftReal*)(data.dptr_ + i * data.shape_[2] * data.shape_[3]), 
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
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4> igrad = in_grad[fft2d::kData].get<xpu, 4, real_t>(s);
        Tensor<xpu, 4> ograd = out_grad[fft2d::kOut].get<xpu, 4, real_t>(s);
        CHECK_EQ(igrad.CheckContiguous(), true);
        CHECK_EQ(ograd.CheckContiguous(), true);
        if (!init_backward_cufft_) {
          Init(igrad.shape_[0], igrad.shape_[1], igrad.shape_[2], igrad.shape_[3], 1);
        }
        #if defined(__CUDACC__)
        const int count = ograd.shape_.Size();
        cudaStream_t stream = Stream<gpu>::GetStream(ograd.stream_);
        dim3 dimGrid((count + FRCNN_NUM_THREADS - 1) / FRCNN_NUM_THREADS);
        dim3 dimBlock(FRCNN_NUM_THREADS);
        RescaleRFFTOutGradKernel<real_t> <<<dimGrid, dimBlock, 0, stream >>>(count, ograd.dptr_, ograd.shape_[3]);
        FRCNN_CUDA_CHECK(cudaPeekAtLastError());
        #endif
        CHECK(0 == (igrad.shape_[0] * igrad.shape_[1]) % param_.batchsize);
        for (int i = 0; i < igrad.shape_[0] * igrad.shape_[1]; i += param_.batchsize) {
          CHECK_EQ(cufftExecC2R(backward_plan, (cufftComplex*)(ograd.dptr_ + i * ograd.shape_[2] * ograd.shape_[3]),
            (cufftReal*)(igrad.dptr_ + i * igrad.shape_[2] * igrad.shape_[3])), CUFFT_SUCCESS);
        }
      }

    private:
      inline void Init(int num, int channel, int rows, int cols, int typ) {
        using namespace mshadow;
        using namespace mshadow::expr;
        int n[2] = { rows, cols };
        // TODO This part may be memory-consuming
        if (num * channel < param_.batchsize) {
          param_.batchsize = num * channel;
        }
        if (0 == typ) {
          CHECK_EQ(cufftPlanMany(&forward_plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, param_.batchsize), CUFFT_SUCCESS);
          init_forward_cufft_ = true;
        }
        else {
          CHECK_EQ(cufftPlanMany(&backward_plan, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, param_.batchsize), CUFFT_SUCCESS);
          init_backward_cufft_ = true;
        }
        
      }
      FFT2DParam param_;
      bool init_forward_cufft_;
      bool init_backward_cufft_;
      cufftHandle forward_plan;
      cufftHandle backward_plan;
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
        CHECK(0 == (dshape[0] * dshape[1]) % param_.batchsize || dshape[0] * dshape[1] < param_.batchsize) 
          << "In FFT2D, the dim[0] * dim[1] must be smaller than or be divide by batchsize. dim[0] = " << dshape[0] << ", dim[1] = " 
          << dshape[1] << ", batchsize = " << param_.batchsize;
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
        return{ out_grad[fft2d::kOut] };
      }

      Operator* CreateOperator(Context ctx) const override;

    private:
      FFT2DParam param_;
    };
#endif  // DMLC_USE_CXX11
  }  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_FFT_GPU_INL_H_