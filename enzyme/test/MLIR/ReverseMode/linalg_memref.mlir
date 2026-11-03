// RUN: %eopt %s --enzyme --simplify-memref-cache --canonicalize --remove-unnecessary-enzyme-ops --add-to-op-to-split --canonicalize | FileCheck %s

// Reverse mode on memref linalg ops: the gradient must be accumulated into the
// shadow of each active input, iterating in reverse over cached values.

#id = affine_map<(i) -> (i)>
#scalar = affine_map<(i) -> ()>

// y[i] = x[i] * x[i]
func.func @square(%x: memref<4xf64>, %y: memref<4xf64>) {
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%x : memref<4xf64>) outs(%y : memref<4xf64>) {
  ^bb0(%a: f64, %b: f64):
    %sq = arith.mulf %a, %a : f64
    linalg.yield %sq : f64
  }
  return
}

func.func @dsquare(%x: memref<4xf64>, %dx: memref<4xf64>, %y: memref<4xf64>, %dy: memref<4xf64>) {
  enzyme.autodiff @square(%x, %dx, %y, %dy) <{activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity = []}> : (memref<4xf64>, memref<4xf64>, memref<4xf64>, memref<4xf64>) -> ()
  return
}

// CHECK-LABEL: func.func private @diffesquare(
// CHECK-SAME:      %[[X:.+]]: memref<4xf64>, %[[DX:.+]]: memref<4xf64>, %[[Y:.+]]: memref<4xf64>, %[[DY:.+]]: memref<4xf64>)
// CHECK:         %[[C1:.+]] = memref.alloc() : memref<4xf64>
// CHECK:         %[[C2:.+]] = memref.alloc() : memref<4xf64>
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[X]] : memref<4xf64>) outs(%[[Y]], %[[C1]], %[[C2]] :
// CHECK:           %[[SQ:.+]] = arith.mulf %[[IN:.+]], %[[IN]] : f64
// CHECK:           linalg.yield %[[SQ]], %[[IN]], %[[IN]] : f64, f64, f64
// CHECK-DAG:     %[[RDY:.+]] = memref.subview %[[DY]][3] [4] [-1]
// CHECK-DAG:     %[[RDX:.+]] = memref.subview %[[DX]][3] [4] [-1]
// CHECK-DAG:     %[[RC1:.+]] = memref.subview %[[C1]][3] [4] [-1]
// CHECK-DAG:     %[[RC2:.+]] = memref.subview %[[C2]][3] [4] [-1]
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[RDY]], %[[RC1]], %[[RC2]] :
// CHECK-SAME:      outs(%[[RDX]] :
// CHECK:         ^bb0(%[[G:.+]]: f64, %[[A:.+]]: f64, %[[B:.+]]: f64, %[[OUT:.+]]: f64):
// CHECK:           %[[GA:.+]] = arith.mulf %{{.+}}, %[[B]]
// CHECK:           %[[GB:.+]] = arith.mulf %{{.+}}, %[[A]]
// CHECK:           %[[SUM:.+]] = arith.addf %{{.+}}, %[[GB]]
// CHECK:           %[[ACC:.+]] = arith.addf %[[OUT]], %[[SUM]]
// CHECK:           linalg.yield %[[ACC]] : f64

// y = sum_i x[i] * x[i]
func.func @sumsq(%x: memref<4xf64>, %y: memref<f64>) {
  linalg.generic {indexing_maps = [#id, #scalar], iterator_types = ["reduction"]}
      ins(%x : memref<4xf64>) outs(%y : memref<f64>) {
  ^bb0(%a: f64, %acc: f64):
    %sq = arith.mulf %a, %a : f64
    %sum = arith.addf %acc, %sq : f64
    linalg.yield %sum : f64
  }
  return
}

func.func @dsumsq(%x: memref<4xf64>, %dx: memref<4xf64>, %y: memref<f64>, %dy: memref<f64>) {
  enzyme.autodiff @sumsq(%x, %dx, %y, %dy) <{activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity = []}> : (memref<4xf64>, memref<4xf64>, memref<f64>, memref<f64>) -> ()
  return
}

// CHECK-LABEL: func.func private @diffesumsq(
// CHECK-SAME:      %[[X:.+]]: memref<4xf64>, %[[DX:.+]]: memref<4xf64>, %[[Y:.+]]: memref<f64>, %[[DY:.+]]: memref<f64>)
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["reduction"]
// CHECK-SAME:      ins(%[[X]] : memref<4xf64>) outs(%[[Y]],
// CHECK:         %[[RDX:.+]] = memref.subview %[[DX]][3] [4] [-1]
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel"]
// CHECK-SAME:      ins(%[[DY]],
// CHECK-SAME:      outs(%[[RDX]] :
// CHECK:           arith.addf %{{.+}}, %{{.+}} fastmath<fast> : f64
// CHECK:           linalg.yield

// C = A B
func.func @matmul(%a: memref<2x3xf64>, %b: memref<3x2xf64>, %c: memref<2x2xf64>) {
  linalg.matmul ins(%a, %b : memref<2x3xf64>, memref<3x2xf64>) outs(%c : memref<2x2xf64>)
  return
}

func.func @dmatmul(%a: memref<2x3xf64>, %da: memref<2x3xf64>, %b: memref<3x2xf64>, %db: memref<3x2xf64>, %c: memref<2x2xf64>, %dc: memref<2x2xf64>) {
  enzyme.autodiff @matmul(%a, %da, %b, %db, %c, %dc) <{activity = [#enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>, #enzyme.activity<enzyme_dup>], ret_activity = []}> : (memref<2x3xf64>, memref<2x3xf64>, memref<3x2xf64>, memref<3x2xf64>, memref<2x2xf64>, memref<2x2xf64>) -> ()
  return
}

// CHECK-LABEL: func.func private @diffematmul(
// CHECK-SAME:      %[[A:.+]]: memref<2x3xf64>, %[[DA:.+]]: memref<2x3xf64>, %[[B:.+]]: memref<3x2xf64>, %[[DB:.+]]: memref<3x2xf64>, %[[C:.+]]: memref<2x2xf64>, %[[DC:.+]]: memref<2x2xf64>)
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[A]], %[[B]] : memref<2x3xf64>, memref<3x2xf64>) outs(%[[C]],
// CHECK-DAG:     %[[RDC:.+]] = memref.subview %[[DC]][1, 1] [2, 2] [-1, -1]
// CHECK-DAG:     %[[RDA:.+]] = memref.subview %[[DA]][1, 2] [2, 3] [-1, -1]
// CHECK-DAG:     %[[RDB:.+]] = memref.subview %[[DB]][2, 1] [3, 2] [-1, -1]
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[RDC]],
// CHECK-SAME:      outs(%[[RDA]] :
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[RDC]],
// CHECK-SAME:      outs(%[[RDB]] :
