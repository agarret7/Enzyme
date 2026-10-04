include "Common.td"

class ImpulseInst<string m> : Inst<m, "impulse">;

def ImpulseDynamicSlice : ImpulseInst<"DynamicSliceOp">;
def ImpulseDynamicUpdateSlice : ImpulseInst<"DynamicUpdateSliceOp">;
def ImpulseReshape : ImpulseInst<"ReshapeOp">;

def ImpulseZero : ConstantFP<"0", "arith", "ConstantOp">;

def ImpulseUpdateSliceShape : GlobalExpr</*needsprimal*/0, /*needsshadow*/0, [{
  builder.getDenseI64ArrayAttr(
      cast<RankedTensorType>(op.getUpdate().getType()).getShape());
}]>;

// dx = zeros_like(x) with dr written into the sliced window.
def : MemoryIdentityOp<"impulse", "DynamicSliceOp", [], [0],
                       (Op $operand, (Variadic<"getStartIndices">):$start_indices),
                       [
                         (ImpulseDynamicUpdateSlice (TypeOf $operand),
                                                    (ImpulseZero $operand),
                                                    (DiffeRet),
                                                    $start_indices),
                         (AssertingInactiveArg)
                       ]>;

// The overwritten window of `operand` receives no gradient; `update` receives that window of dr.
def : MemoryIdentityOp<"impulse", "DynamicUpdateSliceOp", [], [0, 1],
                       (Op $operand, $update, (Variadic<"getStartIndices">):$start_indices),
                       [
                         (ImpulseDynamicUpdateSlice (TypeOf $operand),
                                                    (DiffeRet),
                                                    (ImpulseZero $update),
                                                    $start_indices),
                         (ImpulseDynamicSlice (TypeOf $update),
                                              (DiffeRet),
                                              $start_indices,
                                              (ImpulseUpdateSliceShape)),
                         (AssertingInactiveArg)
                       ]>;

def : ReadOnlyIdentityOp<"impulse", "ReshapeOp", [0], (Op $input),
                         [(ImpulseReshape (TypeOf $input), (DiffeRet))]>;

// Forward mode only; the reverse rule is hand-written in ImpulseAutoDiffOpInterfaceImpl.cpp.
def : ReadOnlyIdentityOp<"impulse", "SliceOp", [0]>;
