// Caller-side stream skew; communication owners do not depend on this helper.
#include <stdint.h>
extern "C" __global__ void witness_spin(uint64_t nanoseconds) {
  uint64_t start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - start < nanoseconds);
}
