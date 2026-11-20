// RUN: if [ %llvmver -ge 12 ]; then %clang -O0 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=0 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=0 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=0 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=0 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O0 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O1 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O2 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1 | %lli - ; fi
// RUN: if [ %llvmver -ge 12 ]; then %clang -O3 %s -S -emit-llvm -o - %loadClangEnzyme -mllvm -enzyme-lapack-copy=1 | %lli - ; fi

// Reverse mode through cblas_dgemm, C = op(A) op(B), whose inputs are overwritten afterwards, so
// the reverse pass reads them from Enzyme's cache: for both layouts and every combination of
// transposes. loss = sum_ij W_ij C_ij, so dop(A) = W op(B)^T and dop(B) = op(A)^T W.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "../test_utils.h"

enum { RowMajor = 101, ColMajor = 102 };
enum { NoTrans = 111, Trans = 112 };

extern int enzyme_dup;
extern int enzyme_const;
void __enzyme_autodiff(void *, ...);

enum { M = 3, N = 4, K = 5 };

// The offset of element (r, c) of a matrix with leading dimension `ld`.
static int at(int layout, int ld, int r, int c) {
  return layout == RowMajor ? r * ld + c : r + c * ld;
}

// A reference cblas_dgemm, C = alpha op(A) op(B) + beta C, which checks its leading dimensions
// as a BLAS library does.
__attribute__((noinline)) void cblas_dgemm(int layout, int ta, int tb, int m, int n, int k,
                                           double alpha, const double *a, int lda,
                                           const double *b, int ldb, double beta, double *c,
                                           int ldc) {
  int arows = ta == NoTrans ? m : k, acols = ta == NoTrans ? k : m;
  int brows = tb == NoTrans ? k : n, bcols = tb == NoTrans ? n : k;
  assert(lda >= (layout == RowMajor ? acols : arows));
  assert(ldb >= (layout == RowMajor ? bcols : brows));
  assert(ldc >= (layout == RowMajor ? n : m));
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      double s = 0;
      for (int p = 0; p < k; p++) {
        double x = ta == NoTrans ? a[at(layout, lda, i, p)] : a[at(layout, lda, p, i)];
        double y = tb == NoTrans ? b[at(layout, ldb, p, j)] : b[at(layout, ldb, j, p)];
        s += x * y;
      }
      double *out = &c[at(layout, ldc, i, j)];
      *out = alpha * s + (beta == 0 ? 0 : beta * *out);
    }
}

// What Enzyme calls to cache a matrix with -enzyme-lapack-copy=1: copies the m x n matrix `a`
// to `b`, both in `layout`.
__attribute__((noinline)) void cblas_dlacpy(int layout, char uplo, int m, int n, const double *a,
                                            int lda, double *b, int ldb) {
  (void)uplo;
  assert(ldb >= (layout == RowMajor ? n : m));
  for (int r = 0; r < m; r++)
    for (int c = 0; c < n; c++)
      b[at(layout, ldb, r, c)] = a[at(layout, lda, r, c)];
}

// What Enzyme calls to scale C's shadow by beta with -enzyme-lapack-copy=1: multiplies the
// m x n matrix `a` by cto / cfrom.
__attribute__((noinline)) void cblas_dlascl(int layout, char type, int kl, int ku, double cfrom,
                                            double cto, int m, int n, double *a, int lda,
                                            int *info) {
  (void)type, (void)kl, (void)ku;
  for (int r = 0; r < m; r++)
    for (int c = 0; c < n; c++)
      a[at(layout, lda, r, c)] *= cto / cfrom;
  *info = 0;
}

// The stored shape of op(X) = rows x cols, transposed by `trans`.
static int stored_rows(int trans, int rows, int cols) { return trans == NoTrans ? rows : cols; }
static int stored_cols(int trans, int rows, int cols) { return trans == NoTrans ? cols : rows; }
static int ld(int layout, int rows, int cols) { return layout == RowMajor ? cols : rows; }

__attribute__((noinline)) static double f(double *a, double *b, double *c, int layout, int ta,
                                          int tb) {
  int lda = ld(layout, stored_rows(ta, M, K), stored_cols(ta, M, K));
  int ldb = ld(layout, stored_rows(tb, K, N), stored_cols(tb, K, N));
  cblas_dgemm(layout, ta, tb, M, N, K, 1.0, a, lda, b, ldb, 0.0, c, ld(layout, M, N));
  double loss = 0;
  for (int i = 0; i < M * N; i++)
    loss += (1.0 + i * 0.25) * c[i];
  // Overwrite both inputs, so the reverse pass needs the cached copies.
  for (int i = 0; i < M * K; i++)
    a[i] = 0;
  for (int i = 0; i < K * N; i++)
    b[i] = 0;
  return loss;
}

static void check(int layout, int ta, int tb) {
  double a[M * K], a0[M * K], da[M * K] = {0}, b[K * N], b0[K * N], db[K * N] = {0};
  double c[M * N] = {0}, dc[M * N] = {0};
  for (int i = 0; i < M * K; i++)
    a[i] = a0[i] = 0.1 + 0.07 * i;
  for (int i = 0; i < K * N; i++)
    b[i] = b0[i] = -0.3 + 0.05 * i;
  __enzyme_autodiff((void *)f, enzyme_dup, a, da, enzyme_dup, b, db, enzyme_dup, c, dc,
                    enzyme_const, layout, enzyme_const, ta, enzyme_const, tb);

  int lda = ld(layout, stored_rows(ta, M, K), stored_cols(ta, M, K));
  int ldb = ld(layout, stored_rows(tb, K, N), stored_cols(tb, K, N));
  // op(A)[i][p], op(B)[p][j] and W[i][j], by their positions in the stored arrays.
#define OPA(x, i, p) (ta == NoTrans ? x[at(layout, lda, i, p)] : x[at(layout, lda, p, i)])
#define OPB(x, p, j) (tb == NoTrans ? x[at(layout, ldb, p, j)] : x[at(layout, ldb, j, p)])
#define W(i, j) (1.0 + at(layout, ld(layout, M, N), i, j) * 0.25)
  for (int i = 0; i < M; i++)
    for (int p = 0; p < K; p++) {
      double expected = 0;
      for (int j = 0; j < N; j++)
        expected += W(i, j) * OPB(b0, p, j);
      APPROX_EQ(OPA(da, i, p), expected, 1e-10);
    }
  for (int p = 0; p < K; p++)
    for (int j = 0; j < N; j++) {
      double expected = 0;
      for (int i = 0; i < M; i++)
        expected += OPA(a0, i, p) * W(i, j);
      APPROX_EQ(OPB(db, p, j), expected, 1e-10);
    }
}

int main() {
  for (int layout = RowMajor; layout <= ColMajor; layout++)
    for (int ta = NoTrans; ta <= Trans; ta++)
      for (int tb = NoTrans; tb <= Trans; tb++)
        check(layout, ta, tb);
  return 0;
}
