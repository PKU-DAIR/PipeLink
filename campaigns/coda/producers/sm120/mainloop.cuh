#pragma once
#include <cutlass/gemm/device/gemm.h>
#include <cuda_bf16.h>
#include <cute/atom/mma_traits_sm80.hpp>
#include "../../abi/coda_tile_args.h"

namespace coda_sm120 {
using Element = cutlass::bfloat16_t;
template<int Rows = 128, int Columns = 256>
struct Configuration {
  using Gemm = cutlass::gemm::device::Gemm<
      Element, cutlass::layout::RowMajor, Element, cutlass::layout::ColumnMajor,
      Element, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<Rows, Columns, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3>;
  using Mma = typename Gemm::GemmKernel::Mma;
  using SharedStorage = typename Mma::SharedStorage;
  using Fragment = typename Mma::FragmentC;
  static_assert(Fragment::kElements == 128);
  static_assert(sizeof(SharedStorage) <= 99 * 1024);
  // Same swizzle-8 raster, with a compact final group (no out-of-grid CTAs).
  CUTLASS_HOST_DEVICE static void origin(int tile, int M, int N, int& m0, int& n0) {
    int mt = M / Rows, nt = N / Columns;
    int group = tile / (8 * nt), first = group * 8;
    int width = mt - first < 8 ? mt - first : 8;
    int within = tile - group * 8 * nt;
    m0 = (first + within % width) * Rows;
    n0 = (within / width) * Columns;
  }
  CUTLASS_HOST_DEVICE static cutlass::MatrixCoord coordinate(int thread, int p) {
    using Atom = cute::MMA_Traits<cute::SM80_16x8x16_F32BF16BF16F32_TN>;
    using Iterator = typename Mma::Operator::IteratorC;
    constexpr int mma_rows = Iterator::Policy::MmaIterations::kRow;
    constexpr int warp_rows = Mma::WarpCount::kM;
    // CUTLASS orders instruction fragments column-major; capture the instruction's
    // lane/value layout from CuTe rather than restating the PTX register map.
    int atom = typename Atom::CLayout{}(thread % 32, p % 4);
    return {thread / 32 % warp_rows * 64 + (p / 4 % mma_rows) * 16 + atom % 16,
            thread / 32 / warp_rows * 64 + (p / (4 * mma_rows)) * 8 + atom / 16};
  }
#if defined(__CUDACC__)
  CUTLASS_DEVICE static void run(coda_tile_args const& args, SharedStorage& shared, Fragment& acc) {
    int t = int(threadIdx.x);
    typename Mma::IteratorA a(typename Mma::IteratorA::Params{cutlass::layout::RowMajor(args.K)},
        static_cast<Element*>(const_cast<void*>(args.a)), {args.M, args.K}, t, {args.m0, 0});
    typename Mma::IteratorB b(typename Mma::IteratorB::Params{cutlass::layout::ColumnMajor(args.K)},
        static_cast<Element*>(const_cast<void*>(args.b)), {args.K, args.N}, t, {0, args.n0});
    Mma mma(shared, t, cutlass::canonical_warp_idx_sync(), t % 32);
    acc.clear();
    mma((args.K + 31) / 32, acc, a, b, acc);
    // Complete every cp.async read before tensor_ready permits arena reuse.
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
  }
#endif
};
using Config = Configuration<>;
} // namespace coda_sm120
