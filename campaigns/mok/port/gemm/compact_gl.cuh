#pragma once

// Owner-private TK gl adapter: preserve TK metadata and TMA encoding while
// replacing descriptor_dict's padded empty tail with a contiguous map array.
// Maps stay in the grid-constant residue, aligned by CUtensorMap itself.
namespace mok_gemm_owner {
template<typename T, int B, int D, int R, int C, typename... Tiles>
struct compact_gl : kittens::gl<T, B, D, R, C> {
  using Base = kittens::gl<T, B, D, R, C>;
  static_assert(sizeof...(Tiles) > 0);
  CUtensorMap maps[sizeof...(Tiles)];

#ifndef KITTENS_NO_HOST
  __host__ compact_gl(T* data, kittens::ducks::gl::make_arg_t<B> b,
      kittens::ducks::gl::make_arg_t<D> d, kittens::ducks::gl::make_arg_t<R> r,
      kittens::ducks::gl::make_arg_t<C> c) : Base(data, b, d, r, c) {
    int i = 0;
    (kittens::detail::tma::create_tensor_map<typename kittens::tma::descriptor<Tiles>::T,
        kittens::tma::descriptor<Tiles>::axis>(&maps[i++], data,
            this->batch(), this->depth(), this->rows(), this->cols()), ...);
  }
#endif
  template<typename U, int Axis>
  __device__ __forceinline__ const CUtensorMap* get_tma() const {
    constexpr int index = [] {
      constexpr bool match[] = {(std::is_same_v<typename kittens::tma::descriptor<Tiles>::T, U> &&
          kittens::tma::descriptor<Tiles>::axis == Axis)...};
      for (int i = 0; i < sizeof...(Tiles); ++i) if (match[i]) return i;
      return -1;
    }();
    static_assert(index >= 0, "Requested tile is absent from the owner descriptor pack");
    return &maps[index];
  }
  template<typename U, int Axis = 2>
  __device__ __forceinline__ void prefetch_tma() const {
    const CUtensorMap* map = get_tma<U, Axis>();
    asm volatile("{prefetch.tensormap [%0];}" :: "l"(reinterpret_cast<uint64_t>(map)) : "memory");
  }
};
static_assert(alignof(CUtensorMap) >= 64);
} // namespace mok_gemm_owner
