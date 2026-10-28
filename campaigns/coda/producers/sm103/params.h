#pragma once
#include <cuda.h>
#include <stdint.h>

struct alignas(128) CodaSm103Params {
  CUtensorMap a;
  CUtensorMap b;
  int32_t num_sms;
  uint32_t stagger_ns;
};
