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

}  // namespace top
}  // namespace nnvm