#pragma once
#include <stdint.h>

typedef struct pipelink_tensor_source { unsigned long long word; } pipelink_tensor_source;

#ifdef __cplusplus
#if defined(__CUDACC__)
#define PIPELINK_SOURCE_INLINE __device__ __forceinline__
#else
#define PIPELINK_SOURCE_INLINE inline
#endif
namespace pl {
PIPELINK_SOURCE_INLINE pipelink_tensor_source source_of(const void* pointer) {
  return {reinterpret_cast<unsigned long long>(pointer)};
}
PIPELINK_SOURCE_INLINE pipelink_tensor_source source_of_tmem(uint32_t address) {
  return {address};
}
template <typename T> PIPELINK_SOURCE_INLINE T* source_pointer(pipelink_tensor_source source) {
  return reinterpret_cast<T*>(source.word);
}
PIPELINK_SOURCE_INLINE uint32_t source_tmem(pipelink_tensor_source source) {
  return static_cast<uint32_t>(source.word);
}
}
#undef PIPELINK_SOURCE_INLINE
#endif
