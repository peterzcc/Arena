/*!
* Copyright (c) 2016 by Contributors
* \file spatial_glimpse-inl.h
* \brief
* \author Xingjian Shi
*/

#ifndef MXNET_OPERATOR_SPATIAL_GLIMPSE_INL_H_
#define MXNET_OPERATOR_SPATIAL_GLIMPSE_INL_H_

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

    namespace spatial_glimpse_enum {
      enum SpatialGlimpseOpInputs { kData, kROI};
      enum SpatialGlimpseOpOutputs { kOut };
    }  // namespace spatial_glimpse_enum

    struct SpatialGlimpseParam : public dmlc::Parameter<SpatialGlimpseParam> {
      TShape output_shape;
      real_t scale;
      DMLC_DECLARE_PARAMETER(SpatialGlimpseParam) {
        DMLC_DECLARE_FIELD(output_shape)
          .set_expect_ndim(2).enforce_nonzero()
          .describe("Size of the spatial_output: (H, W)");
        DMLC_DECLARE_FIELD(scale)
          .set_default(1.0f)
          .describe("Scale multiplication of the glimpse.");
      }
    };

    template<typename xpu>
    class SpatialGlimpseOp : public Operator {
    public:
      explicit SpatialGlimpseOp(SpatialGlimpseParam p) {
        this->param_ = p;
      }

      virtual void Forward(const OpContext &ctx,
        const std::vector<TBlob> &in_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &out_data,
        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(req[spatial_glimpse_enum::kOut], kWriteTo);
        CHECK_EQ(in_data.size(), 2);
        CHECK_EQ(out_data.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4> data = in_data[spatial_glimpse_enum::kData].get<xpu, 4, real_t>(s);
        Tensor<xpu, 2> roi = in_data[spatial_glimpse_enum::kROI].get<xpu, 2, real_t>(s);
        Tensor<xpu, 4> out = out_data[spatial_glimpse_enum::kOut].get<xpu, 4, real_t>(s);
        CHECK(roi.shape_[1] == 4)
          << "Spatial Glimpse only supports roi shape: (nroi, 4)";
        CHECK(out.shape_[0] == data.shape_[0] && data.shape_[0] == roi.shape_[0])
          << "The batchsize of all inputs must be the same";
        index_t dst_height = out.shape_[2];
        index_t dst_width = out.shape_[3];
        Assign(out, req[spatial_glimpse_enum::kOut],
          bilinear_resample(data, roi, dst_height, dst_width, param_.scale));
      }

      virtual void Backward(const OpContext &ctx,
        const std::vector<TBlob> &out_grad,
        const std::vector<TBlob> &in_data,
        const std::vector<TBlob> &out_data,
        const std::vector<OpReqType> &req,
        const std::vector<TBlob> &in_grad,
        const std::vector<TBlob> &aux_args) {
        /*
        using namespace mshadow;
        using namespace mshadow::expr;
        CHECK_EQ(out_grad.size(), 1);
        CHECK_EQ(in_data.size(), 2);
        CHECK_EQ(out_data.size(), 1);
        CHECK_EQ(req.size(), 1);
        CHECK_EQ(in_grad.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();
        Tensor<xpu, 4> grad = out_grad[spatial_glimpse_enum::kOut].get<xpu, 4, real_t>(s);
        Tensor<xpu, 4> data = in_data[spatial_glimpse_enum::kData].get<xpu, 4, real_t>(s);
        Tensor<xpu, 2> roi = in_data[spatial_glimpse_enum::kROI].get<xpu, 2, real_t>(s);
        Tensor<xpu, 4> output_data = out_data[spatial_glimpse_enum::kOut].get<xpu, 4, real_t>(s);
        Tensor<xpu, 4> input_grad = in_grad[spatial_glimpse_enum::kData].get<xpu, 4, real_t>(s);

        Assign(input_grad, req[spatial_glimpse_enum::kData],
          bilinear_resample_grad(grad, data, roi, param_.scale));
        */
      }

    private:
      SpatialGlimpseParam param_;
    };  // class SpatialGlimpseOp

    template<typename xpu>
    Operator* CreateOp(SpatialGlimpseParam param);


#if DMLC_USE_CXX11
    class SpatialGlimpseProp : public OperatorProperty {
    public:
      std::vector<std::string> ListArguments() const override {
        return{ "data", "roi"};
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
        CHECK_EQ(in_shape->size(), 2);
        const TShape &data_shape = (*in_shape)[0];
        const TShape &roi_shape = (*in_shape)[1];
        CHECK_EQ(data_shape.ndim(), 4) << \
          "SpatialGlimpse: Input data should be 4D in (batch, channel, H, W)";
        CHECK_EQ(roi_shape.ndim(), 2) << \
          "SpatialGlimpse: ROI data should be 2D in (batch, 4), each element is (cx, cy, sx, sy)";
        TShape oshape = data_shape;
        oshape[2] = param_.output_shape[0];
        oshape[3] = param_.output_shape[1];
        CHECK(oshape[2] > 0 && oshape[3] > 0) << "Spatial Glimpse: Output size must be bigger or equal to 1";
        out_shape->clear();
        out_shape->push_back(oshape);
        return true;
      }

      OperatorProperty* Copy() const override {
        SpatialGlimpseProp *prop_sym = new SpatialGlimpseProp();
        prop_sym->param_ = this->param_;
        return prop_sym;
      }

      std::string TypeString() const override {
        return "SpatialGlimpse";
      }

      std::vector<int> DeclareBackwardDependency(
        const std::vector<int> &out_grad,
        const std::vector<int> &in_data,
        const std::vector<int> &out_data) const override {
        //return{ out_grad[spatial_glimpse_enum::kOut], in_data[spatial_glimpse_enum::kData], in_data[spatial_glimpse_enum::kROI] };
        return{ };
      }

      Operator* CreateOperator(Context ctx) const override;

    private:
      SpatialGlimpseParam param_;
    };  // class SpatialGlimpseProp
#endif  // DMLC_USE_CXX11
  }  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SPATIAL_GLIMPSE_INL_H_
