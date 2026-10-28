#pragma once

// Set for both host annotation capture and device compilation.
#ifndef CODA_PROFILE
#error "Define CODA_PROFILE as 103, 90, or 120"
#endif
#if CODA_PROFILE == 103
#define CODA_THREADS 256
#define CODA_FIRST_DATA_THREAD 0
#define CODA_CTA_GROUP 2
#define CODA_CLUSTER_X 2
#define CODA_IMPORT_SCRATCH_BYTES 0
#elif CODA_PROFILE == 90
#define CODA_THREADS 384
#define CODA_FIRST_DATA_THREAD 128
#define CODA_CTA_GROUP 1
#define CODA_CLUSTER_X 1
#define CODA_IMPORT_SCRATCH_BYTES 16384
#elif CODA_PROFILE == 120
#define CODA_THREADS 256
#define CODA_FIRST_DATA_THREAD 0
#define CODA_CTA_GROUP 1
#define CODA_CLUSTER_X 1
#define CODA_IMPORT_SCRATCH_BYTES 16384
#else
#error "Unsupported CODA_PROFILE"
#endif
#define CODA_OUTPUT_VECTOR_BF16 8

namespace coda_epilogue {
constexpr int chunks = 8;
constexpr int fragment_words = 16;
constexpr int data_threads = 256;
__host__ __device__ constexpr int row(int t) {
#if CODA_PROFILE == 103
  return t % 128;
#else
  return t / 2;
#endif
}
__host__ __device__ constexpr int half_row(int t) {
#if CODA_PROFILE == 103
  return t / 128;
#else
  return t % 2;
#endif
}
}
