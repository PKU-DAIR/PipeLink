#include "../../producers/sm120/mainloop.cuh"

// Exact gate finalist, including CUTLASS's vectorized bf16 output epilogue.
extern "C" int coda_sm120_gate_reference(void const* a, void const* b, void* d,
                                         int M, int N, int K, cudaStream_t stream) {
  using C = coda_sm120::Configuration<256, 128>;
  using E = coda_sm120::Element;
  C::Gemm gemm;
  C::Gemm::Arguments arguments({M,N,K},
      {static_cast<E const*>(a),K}, {static_cast<E const*>(b),K},
      {static_cast<E*>(d),N}, {static_cast<E*>(d),N}, {1.0f,0.0f});
  auto status = gemm.initialize(arguments, nullptr, stream);
  if (status != cutlass::Status::kSuccess) return int(status);
  return int(gemm(stream));
}
