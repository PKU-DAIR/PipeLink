#include <pipelink/stage.hpp>

struct alignas(16) Storage { unsigned words[4]; };
PL_CLAIM("pipelink_work", "smem", 16, 16,
         pl::during{"value", "work", "ready"}, "x");
PL_CLAIM_SMEM("pipelink_ready", Storage, pl::during{"value", "ready", "done"});
PL_CLAIM_EX(pl::claim("pipelink_after", "smem").extent(16).align(16)
    .lifetime(pl::during{"value", "done"}).roles({"x"}));
PL_CLAIM("pipelink_kernel", "smem", 16, 16, "kernel");
PL_CLAIM_SMEM("pipelink_phase", Storage, "phase:compute", "x");
PL_CLAIM_EX(pl::claim("pipelink_override", "smem").extent(16).align(16)
    .lifetime(pl::during{"value", "work", "ready"}).lifetime("kernel"));
PL_SITE(completes_once, "x", pl::all_threads, "work",
        pl::guarantees{"enter_once", "single_completion"}, pl::strings{"ready"});
