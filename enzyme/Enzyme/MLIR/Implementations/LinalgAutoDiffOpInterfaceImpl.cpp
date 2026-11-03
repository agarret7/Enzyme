//===- LinalgAutoDiffOpInterfaceImpl.cpp - Interface external model -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Passes/RemovalUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include <functional>

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {

Value invertMemref(Value inp, OpBuilder &builder, Location loc) {
  MemRefType iType = cast<MemRefType>(inp.getType());
  SmallVector<Value> dims;
  SmallVector<Value> dimSubOnes;
  SmallVector<Value> strides;
  Value negOne = arith::ConstantIndexOp::create(builder, loc, -1);
  int shapeDim = iType.getShape().size();
  for (int i = 0; i < shapeDim; i++) {
    Value dim = memref::DimOp::create(builder, loc, inp, i);
    dims.push_back(dim);
    auto dimSubOne = arith::AddIOp::create(builder, loc, dim, negOne);
    dimSubOnes.push_back(dimSubOne);
    strides.push_back(negOne);
  }
  Value view =
      memref::SubViewOp::create(builder, loc, inp, ValueRange(dimSubOnes),
                                ValueRange(dims), ValueRange(strides));
  return view;
}

SmallVector<AffineMap> getIndexingMapsArray(enzyme::GenericAdjointOp &op) {
  auto attr = op.getIndexingMapsAttr();
  SmallVector<AffineMap> indexingMaps;
  for (auto map : attr.getValue()) {
    indexingMaps.push_back(cast<AffineMapAttr>(map).getValue());
  }
  return indexingMaps;
}

template <typename T_>
struct GenericOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          GenericOpInterfaceReverse<T_>, T_> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    if (!linalgOp.hasPureBufferSemantics()) {
      llvm::errs() << "Linalg op with tensor semantics not yet supported\n";
      return failure();
    }

    linalg::LinalgOp newOp =
        cast<linalg::LinalgOp>(gutils->getNewFromOriginal(linalgOp));

    // Replace the op by a linalg.generic op if necessary
    IRRewriter rewriter(builder.getContext(), builder.getListener());
    // generalizeNamedOp creates the generic and replaces the named op itself,
    // so only the mapping from the original op needs updating.
    rewriter.setInsertionPoint(newOp);
    auto failiureOrLinalgOp = generalizeNamedOp(rewriter, newOp);
    if (!failed(failiureOrLinalgOp)) {
      newOp = failiureOrLinalgOp.value();
      gutils->originalToNewFnOps[op] = newOp;
    }

    auto cacheBuilder = OpBuilder(newOp, builder.getListener());

    // Calculate the iteration domain
    AffineMap aMap = newOp.getShapesToLoopsMap();
    SmallVector<Value> dims;
    for (OpOperand *input : newOp.getDpsInputOperands()) {
      auto shape = cast<MemRefType>(input->get().getType()).getShape();
      for (unsigned i = 0; i < shape.size(); i++) {
        auto dimI =
            arith::ConstantIndexOp::create(cacheBuilder, op->getLoc(), i);
        auto dim = memref::DimOp::create(cacheBuilder, op->getLoc(),
                                         input->get(), dimI);
        dims.push_back(dim);
      }
    }
    for (Value output : newOp.getDpsInits()) {
      auto shape = cast<MemRefType>(output.getType()).getShape();
      for (unsigned i = 0; i < shape.size(); i++) {
        auto dimI =
            arith::ConstantIndexOp::create(cacheBuilder, op->getLoc(), i);
        auto dim =
            memref::DimOp::create(cacheBuilder, op->getLoc(), output, dimI);
        dims.push_back(dim);
      }
    }

    SmallVector<Value> iterationDomains;
    SmallVector<int64_t> shapes;
    for (unsigned int i = 0; i < aMap.getNumResults(); i++) {
      AffineMap subMap = aMap.getSubMap({i});
      Value domain = affine::AffineApplyOp::create(cacheBuilder, op->getLoc(),
                                                   subMap, ValueRange(dims));
      iterationDomains.push_back(domain);
      shapes.push_back(ShapedType::kDynamic);
    }
    //

    SmallVector<Value> inputs, outputs;
    SmallVector<AffineMap> indexingMaps;
    SmallVector<utils::IteratorType> iteratorTypes{
        linalgOp.getNumLoops(), utils::IteratorType::parallel};

    for (OpOperand &output : linalgOp.getDpsInitsMutable()) {
      if (gutils->isConstantValue(output.get())) {
        continue;
      }
      indexingMaps.push_back(linalgOp.getMatchingIndexingMap(&output));
      Value out = gutils->invertPointerM(output.get(), builder);
      Value view = invertMemref(out, builder, op->getLoc());
      outputs.push_back(view);
    }

    for (OpOperand *input : linalgOp.getDpsInputOperands()) {
      if (gutils->isConstantValue(input->get())) {
        continue;
      }
      indexingMaps.push_back(linalgOp.getMatchingIndexingMap(input));
      Value inp = gutils->invertPointerM(input->get(), builder);
      Value view = invertMemref(inp, builder, op->getLoc());
      inputs.push_back(view);
    }

    ArrayAttr indexingMapsArrayAttr =
        builder.getAffineMapArrayAttr(indexingMaps);
    ArrayAttr iteratorTypesArrayAttr =
        builder.getArrayAttr(llvm::to_vector(llvm::map_range(
            iteratorTypes, [&](utils::IteratorType iter) -> mlir::Attribute {
              return linalg::IteratorTypeAttr::get(builder.getContext(), iter);
            })));
    auto adjoint = enzyme::GenericAdjointOp::create(
        builder, op->getLoc(), TypeRange(), ValueRange(outputs),
        ValueRange(inputs), indexingMapsArrayAttr, iteratorTypesArrayAttr,
        StringAttr(), StringAttr());

    Region *newOpRegion = newOp.getBlock()->getParent();
    Region *adjointRegion = &adjoint.getRegion();
    int numInputsAdjoint = adjoint.getInputs().size();
    Location loc = op->getLoc();
    int numCaches = 0;
    SmallVector<Value> pushCaches;

    // The adjoint body's arguments are the cotangents of the active outputs,
    // then any cached values (inserted by `hook`), then one accumulator per
    // active input. Create the cotangents and accumulators up front so the
    // hook inserts at a valid index.
    Block *oBB = linalgOp.getBlock();
    Block *body = new Block();
    adjointRegion->push_back(body);
    for (OpOperand &output : linalgOp.getDpsInitsMutable())
      if (!gutils->isConstantValue(output.get()))
        body->addArgument(linalgOp.getMatchingBlockArgument(&output).getType(),
                          loc);
    for (OpOperand *input : linalgOp.getDpsInputOperands())
      if (!gutils->isConstantValue(input->get()))
        body->addArgument(linalgOp.getMatchingBlockArgument(input).getType(),
                          loc);

    auto hook = [newOpRegion, adjointRegion, loc, &numCaches = numCaches,
                 numInputsAdjoint, &pushCaches = pushCaches](Type t) {
      OpBuilder builder(newOpRegion);
      Value pushCache = enzyme::InitOp::create(builder, loc, t);
      pushCaches.push_back(pushCache);
      newOpRegion->addArgument(t, loc);

      Value popCache =
          adjointRegion->insertArgument(numInputsAdjoint + numCaches, t, loc);
      numCaches++;
      return std::make_pair(pushCache, popCache);
    };

    // Differentiate the body in the style of scf.for's reverse rule: seed the
    // yielded values' gradients from the cotangent arguments, visit the body
    // in reverse, then hand each active input's gradient to `enzyme.addTo`,
    // which accumulates it into that input's shadow.
    gutils->registerCacheCreatorHook(hook);
    OpBuilder bodyBuilder(body, body->end());
    mlir::enzyme::localizeGradients(bodyBuilder, gutils, oBB);

    Operation *oTerm = oBB->getTerminator();
    unsigned argIdx = 0;
    for (auto [idx, output] : llvm::enumerate(linalgOp.getDpsInitsMutable())) {
      if (gutils->isConstantValue(output.get()))
        continue;
      Value yielded = oTerm->getOperand(idx);
      if (!gutils->isConstantValue(yielded))
        gutils->addToDiffe(yielded, body->getArgument(argIdx), bodyBuilder);
      argIdx++;
    }

    bool valid = true;
    for (auto it = std::next(oBB->rbegin()); it != oBB->rend(); ++it)
      valid &= gutils->Logic.visitChild(&*it, bodyBuilder, gutils).succeeded();
    gutils->deregisterCacheCreatorHook(hook);
    if (!valid)
      return failure();

    SmallVector<Value> inputGradients;
    for (OpOperand *input : linalgOp.getDpsInputOperands()) {
      if (gutils->isConstantValue(input->get()))
        continue;
      BlockArgument arg = linalgOp.getMatchingBlockArgument(input);
      inputGradients.push_back(
          gutils->isConstantValue(arg)
              ? cast<AutoDiffTypeInterface>(arg.getType())
                    .createNullValue(bodyBuilder, loc)
              : gutils->diffe(arg, bodyBuilder));
    }
    enzyme::AddToOp::create(bodyBuilder, loc, inputGradients);

    auto newOpYield = cast<linalg::YieldOp>(
        cast<linalg::GenericOp>(newOp).getBodyRegion().front().getTerminator());
    for (Value pc : pushCaches) {
      newOpYield.getValuesMutable().append(pc);
    }

    auto yieldOp = cast<enzyme::AddToOp>(body->getTerminator());

    OpBuilder builderAdd(yieldOp);

    auto newIndexingMaps = newOp.getIndexingMapsArray();
    auto indexingMapsAdjoint = getIndexingMapsArray(adjoint);
    for (int i = 0; i < numCaches; i++) {
      Value cacheArg = body->getArgument(outputs.size() + i);

      Type ct = cacheArg.getType();
      Type type = MemRefType::get(shapes, ct);
      auto alloc = memref::AllocOp::create(cacheBuilder, op->getLoc(), type,
                                           ValueRange(iterationDomains));
      Value cache = gutils->initAndPushCache(alloc, cacheBuilder);
      // TODO use higher level API
      alloc->setAttr(alloc.getOperandSegmentSizesAttrName(),
                     cacheBuilder.getDenseI32ArrayAttr(
                         {static_cast<int32_t>(iterationDomains.size()), 0}));

      cast<linalg::GenericOp>(newOp).getOutputsMutable().append(
          ValueRange({alloc}));
      newIndexingMaps.push_back(AffineMap::getMultiDimIdentityMap(
          iterationDomains.size(), cacheBuilder.getContext()));

      builderAdd.setInsertionPoint(adjoint);
      Value retrievedValue = gutils->popCache(cache, builderAdd);
      retrievedValue = invertMemref(retrievedValue, builderAdd, op->getLoc());
      adjoint.getInputsMutable().append(ValueRange({retrievedValue}));
      indexingMapsAdjoint.insert(
          indexingMapsAdjoint.begin() + numInputsAdjoint + i,
          AffineMap::getMultiDimIdentityMap(iterationDomains.size(),
                                            builderAdd.getContext()));
    }
    SmallVector<Attribute> indexingMapsAttr;
    SmallVector<Attribute> indexingMapsAttrAdjoint;
    for (auto &map : newIndexingMaps) {
      indexingMapsAttr.push_back(AffineMapAttr::get(map));
    }
    for (auto &map : indexingMapsAdjoint) {
      indexingMapsAttrAdjoint.push_back(AffineMapAttr::get(map));
    }
    cast<linalg::GenericOp>(newOp).setIndexingMapsAttr(
        cacheBuilder.getArrayAttr(indexingMapsAttr));
    adjoint->setAttr(adjoint.getIndexingMapsAttrName(),
                     builder.getArrayAttr(indexingMapsAttrAdjoint));
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return SmallVector<Value>();
  }

  LogicalResult createShadowValues(Operation *op, OpBuilder &builder,
                                   MGradientUtilsReverse *gutils) const {
    return success();
  }
};

class GenericFwd
    : public AutoDiffOpInterface::ExternalModel<GenericFwd, linalg::GenericOp> {
public:
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {

    auto op = cast<linalg::GenericOp>(orig);

    // For all active results, add shadow types.
    // For now, assuming all results are relevant.
    Operation *newOp = gutils->getNewFromOriginal(op);
    SmallVector<Type> newOpResultTypes;
    newOpResultTypes.reserve(op->getNumResults() * 2);
    for (auto &&[result, init] :
         llvm::zip_equal(op->getResults(), op.getOutputs())) {
      newOpResultTypes.push_back(result.getType());
      if (gutils->isConstantValue(result) && gutils->isConstantValue(init)) {
        continue;
      }
      auto typeIface = dyn_cast<AutoDiffTypeInterface>(result.getType());
      if (!typeIface) {
        op->emitError() << " AutoDiffTypeInterface not implemented for "
                        << result.getType() << "\n";
        return failure();
      }
      newOpResultTypes.push_back(typeIface.getShadowType(gutils->width));
    }

    SmallVector<Value> newInputs;
    SmallVector<Value> newOutputs;
    SmallVector<AffineMap> indexingMaps;
    {
      size_t idx = 0;
      for (Value operand : op.getInputs()) {
        newInputs.push_back(gutils->getNewFromOriginal(operand));
        indexingMaps.push_back(op.getIndexingMapsArray()[idx]);
        if (!gutils->isConstantValue(operand)) {
          newInputs.push_back(gutils->invertPointerM(operand, builder));
          indexingMaps.push_back(op.getIndexingMapsArray()[idx]);
        }
        idx++;
      }
      for (auto &&[operand, res, oarg] :
           llvm::zip_equal(op.getOutputs(), op->getResults(),
                           op.getRegion().front().getArguments().slice(
                               op.getInputs().size()))) {
        newOutputs.push_back(gutils->getNewFromOriginal(operand));
        indexingMaps.push_back(op.getIndexingMapsArray()[idx]);
        bool shadow = false;
        if (!gutils->isConstantValue(operand)) {
          shadow = true;
          newOutputs.push_back(gutils->invertPointerM(operand, builder));
          indexingMaps.push_back(op.getIndexingMapsArray()[idx]);
        } else if (!gutils->isConstantValue(res)) {
          auto typeIface = dyn_cast<AutoDiffTypeInterface>(operand.getType());
          shadow = true;
          newOutputs.push_back(
              typeIface.createNullValue(builder, operand.getLoc()));
          indexingMaps.push_back(op.getIndexingMapsArray()[idx]);
        }

        if (shadow && gutils->isConstantValue(oarg)) {
          auto typeIface = dyn_cast<AutoDiffTypeInterface>(oarg.getType());
          auto newBA = cast<BlockArgument>(gutils->getNewFromOriginal(oarg));
          newBA.getOwner()->insertArgument(newBA.getArgNumber() + 1,
                                           typeIface.getShadowType(),
                                           newBA.getLoc());
        }

        idx++;
      }
    }
    // We are assuming the op can forward additional operands, listed
    // immediately after the original operands, to the same regions.
    // ^^
    // Our interface guarantees this.
    // We also assume that the region-holding op returns all of the values
    // yielded by terminators, and only those values.

    auto replacement = linalg::GenericOp::create(
        builder, op.getLoc(), newOpResultTypes, newInputs, newOutputs,
        indexingMaps, op.getIteratorTypesArray(),
        /*doc*/ "",
        /*libraryCall*/ "");

    assert(replacement->getNumResults() == newOpResultTypes.size());
    for (auto &&[region, replacementRegion] :
         llvm::zip(newOp->getRegions(), replacement->getRegions())) {
      replacementRegion.takeBody(region);
    }

    // Inject the mapping for the new results into GradientUtil's shadow
    // table.
    SmallVector<Value> reps;
    size_t idx = 0;
    for (OpResult r : op->getResults()) {
      // TODO only if used
      reps.push_back(replacement->getResult(idx));
      idx++;
      if (!gutils->isConstantValue(r)) {
        auto inverted = gutils->invertedPointers.lookupOrNull(r);
        assert(inverted);
        gutils->invertedPointers.map(r, replacement->getResult(idx));
        inverted.replaceAllUsesWith(replacement->getResult(idx));
        gutils->erase(inverted.getDefiningOp());
        idx++;
      }
    }

    // Differentiate body.
    for (auto &origRegion : op->getRegions()) {
      for (auto &origBlock : origRegion) {
        for (Operation &o : origBlock) {
          if (failed(gutils->visitChild(&o))) {
            return failure();
          }
        }
      }
    }

    // Replace all uses of original results
    gutils->replaceOrigOpWith(op, reps);
    gutils->erase(newOp);
    gutils->originalToNewFnOps[op] = replacement;

    return success();
  }
};

// Describes how data flows through a structured linalg op for activity
// analysis. Input block arguments are read from the input operands; output
// block arguments carry the `outs` operand's initial value and, across
// iterations of a reduction, the previous `linalg.yield`; and each yielded
// value is written to its `outs` operand (and becomes the result, for tensors).
template <typename T_>
struct LinalgOpADDataFlow
    : public ADDataFlowOpInterface::ExternalModel<LinalgOpADDataFlow<T_>, T_> {
  SmallVector<Value> getPotentialIncomingValuesRes(Operation *op,
                                                   OpResult res) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    unsigned idx = res.getResultNumber();
    return {linalgOp.getDpsInitOperand(idx)->get(),
            linalgOp.getBlock()->getTerminator()->getOperand(idx)};
  }

  SmallVector<Value> getPotentialIncomingValuesArg(Operation *op,
                                                   BlockArgument arg) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    unsigned numInputs = linalgOp.getNumDpsInputs();
    unsigned idx = arg.getArgNumber();
    if (idx < numInputs)
      return {linalgOp.getDpsInputOperand(idx)->get()};
    idx -= numInputs;
    return {linalgOp.getDpsInitOperand(idx)->get(),
            linalgOp.getBlock()->getTerminator()->getOperand(idx)};
  }

  SmallVector<Value> getPotentialTerminatorUsers(Operation *op, Operation *term,
                                                 Value val) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    unsigned numInputs = linalgOp.getNumDpsInputs();
    SmallVector<Value> users;
    for (auto [idx, yielded] : llvm::enumerate(term->getOperands())) {
      if (yielded != val)
        continue;
      users.push_back(linalgOp.getDpsInitOperand(idx)->get());
      users.push_back(linalgOp.getBlock()->getArgument(numInputs + idx));
      if (idx < op->getNumResults())
        users.push_back(op->getResult(idx));
    }
    return users;
  }
};

#include "Implementations/LinalgDerivatives.inc"
} // namespace

template <typename... Ts> void attachAllInterfaces(MLIRContext *context) {
  (Ts::template attachInterface<GenericOpInterfaceReverse<Ts>>(*context), ...);
  (Ts::template attachInterface<LinalgOpADDataFlow<Ts>>(*context), ...);
}

void mlir::enzyme::registerLinalgDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, linalg::LinalgDialect *) {
    registerInterfaces(context);
    linalg::GenericOp::attachInterface<GenericFwd>(*context);
    attachAllInterfaces<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >(context);
  });
}
