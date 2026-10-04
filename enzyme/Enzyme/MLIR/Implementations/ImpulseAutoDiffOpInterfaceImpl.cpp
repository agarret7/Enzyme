//===- ImpulseAutoDiffOpInterfaceImpl.cpp - Interface external model ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the Impulse dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/ImpulseDerivatives.inc"

// impulse.slice has no pad counterpart, so its adjoint writes dr into zeros
// with dynamic_update_slice. Only unit strides are supported.
class AutoDiffSliceRev
    : public ReverseAutoDiffOpInterface::ExternalModel<AutoDiffSliceRev,
                                                       impulse::SliceOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto op = cast<impulse::SliceOp>(orig);
    if (llvm::any_of(op.getStrides(), [](int64_t s) { return s != 1; })) {
      orig->emitError() << "reverse mode for impulse.slice with non-unit "
                           "strides is not implemented";
      return failure();
    }

    Location loc = op.getLoc();
    Type operandType = op.getOperand().getType();
    Value dr = gutils->diffe(op, builder);
    gutils->zeroDiffe(op, builder);

    auto indexType = RankedTensorType::get({}, builder.getI64Type());
    SmallVector<Value> starts;
    for (int64_t start : op.getStartIndices())
      starts.push_back(arith::ConstantOp::create(
          builder, loc, indexType,
          DenseElementsAttr::get(indexType, builder.getI64IntegerAttr(start))));

    Value zeros = cast<AutoDiffTypeInterface>(operandType)
                      .createNullValue(builder, loc);
    Value dx = impulse::DynamicUpdateSliceOp::create(builder, loc, operandType,
                                                     zeros, dr, starts);
    gutils->addToDiffe(op.getOperand(), dx, builder);
    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    return success();
  }
};
} // namespace

void mlir::enzyme::registerImpulseDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, impulse::ImpulseDialect *) {
    registerInterfaces(context);
    impulse::SliceOp::attachInterface<AutoDiffSliceRev>(*context);
  });
}
