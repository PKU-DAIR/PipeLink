// Full-only completion signal: one phase, with no data buffer or empty barrier.
#include <pipelink/stage.hpp>

#ifndef SIGNAL_SCOPE
#define SIGNAL_SCOPE "cta"
#endif
#ifndef SIGNAL_ARRIVAL_COUNT
#define SIGNAL_ARRIVAL_COUNT 1
#endif
#ifndef SIGNAL_CLUSTER_X
#define SIGNAL_CLUSTER_X 1
#endif

struct Params { unsigned* arrivals; };

extern "C" __device__ int pipelink_signal_full();
extern "C" __device__ void sm90_signal_producer_work(const void* raw, unsigned) {
    if ((threadIdx.x & 31u) >= SIGNAL_ARRIVAL_COUNT) return;
    const auto* params = static_cast<const Params*>(raw);
    params->arrivals[blockIdx.x * SIGNAL_ARRIVAL_COUNT + (threadIdx.x & 31u)] = 1;
    extern __shared__ __align__(16) unsigned char signal_storage[];
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(
        signal_storage + 16u * static_cast<unsigned>(pipelink_signal_full())));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                 : : "r"(address) : "memory");
}

PL_STAGE("org.sm90.signal.producer", "1.0.0");
PL_PARAMS(Params, arrivals);
PL_ROLE(producer, sm90_signal_producer_work, 1);
PL_CLAIM_EX(pl::claim("pipelink_signal_full", "mbarrier").extent(1));
PL_ENDPOINT(completion, pl::signal_endpoint("completion", pl::produce, "producer")
    .scope(SIGNAL_SCOPE)
    .protocol(pl::mbarrier{
        pl::side{"pipelink_signal_full", pl::arrive{"producer", SIGNAL_ARRIVAL_COUNT}, 0},
        std::nullopt, {"producer", 0, "dispatch"}, "external"})
    .bind("full", "pipelink_signal_full"));
PL_TOPOLOGY(pl::topology{1,
    pl::cluster{{SIGNAL_CLUSTER_X, 1, 1}, pl::launch{1, {SIGNAL_CLUSTER_X, 1, 1}}},
    false, "static", {}});
