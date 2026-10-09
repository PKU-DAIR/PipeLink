#include <pipelink/stage.hpp>

#if defined(MISSING_PROTOCOL)
PL_ENDPOINT(incomplete, pl::endpoint("incomplete", pl::produce, "compute")
    .space("smem").scope("cta").proxy("generic").tile("f32", {4})
    .layout(pl::affine{{4}, {1}, 32}));
#elif defined(EMPTY_REG_VALUES)
PL_CLAIM_REGS("pipelink_regs", "compute_group", {}, "compute");
#elif defined(MIXED_BARRIER_SIDE)
constexpr auto bad_barrier() {
  pl::mbarrier barrier{
      pl::member{"a", pl::tile},
      pl::side{"pipelink_empty", pl::arrive("compute", 32), 0},
      pl::init{"compute", 0, "stage"}};
  barrier.full.slots = "pipelink_full";
  return barrier;
}
PL_ENDPOINT(mixed, pl::endpoint("mixed", pl::produce, "compute")
    .space("smem").scope("cta").proxy("generic").tile("f32", {4})
    .layout(pl::affine{{4}, {1}, 32}).protocol(bad_barrier()));
#endif
