/*!
 *  Copyright (c) 2017 by Contributors
 * \file object_dectection.cc
 * \brief Property def of object detection related operators.
 */

#include <tvm/expr.h>
#include <tvm/packed_func_ext.h>
#include <nnvm/op.h>
#include <nnvm/top/nn.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/compiler/op_attr_types.h>
#include "../op_common.h"
#include "../elemwise_op_common.h"

namespace nnvm {
namespace top {
using compiler::FTVMCompute;
using tvm::Tensor;
using tvm::Array;

DMLC_REGISTER_PARAMETER(MultiBoxPriorParam);

bool MultiBoxPriorShape(const NodeAttrs& attrs,
                        std::vector<TShape> *in_attrs,
                        std::vector<TShape> *out_attrs) {
  const MultiBoxPriorParam& param = nnvm::get<MultiBoxPriorParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U) << "Inputs: [data]" << in_attrs->size();
  CHECK_GE((*in_attrs)[0].ndim(), 4U) << "Input data should be 4D: [batch, channel, height, width]";
  int in_height = (*in_attrs)[0][2];
  CHECK_GT(in_height, 0) << "Input height should > 0";
  int in_width = (*in_attrs)[0][3];
  CHECK_GT(in_width, 0) << "Input width should > 0";
  // since input sizes are same in each batch, we could share MultiBoxPrior
  TShape oshape = TShape(3);
  int num_sizes = param.sizes.ndim();
  int num_ratios = param.ratios.ndim();
  oshape[0] = 1;
  oshape[1] = in_height * in_width * (num_sizes + num_ratios - 1);
  oshape[2] = 4;
  CHECK_EQ(param.steps.ndim(), 2) << "Step ndim must be 2: (step_y, step_x)";
  CHECK_GE(param.steps[0] * param.steps[1], 0) << "Must specify both step_y and step_x";
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(multibox_prior)
  .describe(R"doc("Generate prior(anchor) boxes from data, sizes and ratios."
)doc" NNVM_ADD_FILELINE)
.set_support_level(1)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<MultiBoxPriorParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<MultiBoxPriorParam>)
.add_arguments(MultiBoxPriorParam::__FIELDS__())
.add_argument("data", "Tensor", "Input data")
.set_attr<FInferShape>("FInferShape", MultiBoxPriorShape)
.set_attr<FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    return std::vector<NodeEntry>{
      MakeNode("zeros_like", n->attrs.name + "_zero_grad",
               {n->inputs[0]}),
      ograds[0]
    };
});

DMLC_REGISTER_PARAMETER(MultiBoxDetectionParam);

bool MultiBoxDetectionShape(const NodeAttrs& attrs,
                            std::vector<TShape> *in_attrs,
                            std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U) << "Inputs: [cls_prob, loc_pred, anchor]";
  TShape cshape = in_attrs->at(0);
  TShape lshape = in_attrs->at(1);
  TShape ashape = in_attrs->at(2);
  CHECK_EQ(cshape.ndim(), 3U) << "Provided: " << cshape;
  CHECK_EQ(lshape.ndim(), 2U) << "Provided: " << lshape;
  CHECK_EQ(ashape.ndim(), 3U) << "Provided: " << ashape;
  CHECK_EQ(cshape[2], ashape[1]) << "Number of anchors mismatch";
  CHECK_EQ(cshape[2] * 4, lshape[1]) << "# anchors mismatch with # loc";
  CHECK_GT(ashape[1], 0U) << "Number of anchors must > 0";
  CHECK_EQ(ashape[2], 4U);
  TShape oshape = TShape(3);
  oshape[0] = cshape[0];
  oshape[1] = ashape[1];
  oshape[2] = 6;  // [id, prob, xmin, ymin, xmax, ymax]
  out_attrs->clear();
  NNVM_ASSIGN_OUTPUT_SHAPE(attrs, *out_attrs, 0, oshape);
  return true;
}

NNVM_REGISTER_OP(multibox_detection)
  .describe(R"doc("Convert multibox detection predictions."
)doc" NNVM_ADD_FILELINE)
.set_support_level(1)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<MultiBoxDetectionParam>)
.set_attr<FGetAttrDict>("FGetAttrDict", ParamGetAttrDict<MultiBoxDetectionParam>)
.add_arguments(MultiBoxDetectionParam::__FIELDS__())
.add_argument("cls_prob", "Tensor", "Class probabilities.")
.add_argument("loc_pred", "Tensor", "Location regression predictions.")
.add_argument("anchor", "Tensor", "Multibox prior anchor boxes")
.set_attr<FInferShape>("FInferShape", MultiBoxDetectionShape)
.set_attr<FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<FGradient>(
  "FGradient", [](const NodePtr& n,
                  const std::vector<NodeEntry>& ograds) {
    return std::vector<NodeEntry>{
      MakeNode("zeros_like", n->attrs.name + "_zero_grad0",
               {n->inputs[0]}),
      MakeNode("zeros_like", n->attrs.name + "_zero_grad1",
               {n->inputs[1]}),
      MakeNode("zeros_like", n->attrs.name + "_zero_grad2",
               {n->inputs[2]})
    };
});

}  // namespace top
}  // namespace nnvm
