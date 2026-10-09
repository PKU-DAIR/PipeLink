#include <pipelink/stage.hpp>
constexpr auto mapper = [offset = 3](int m, pl::bound base) {
  return std::array<int, 2>{m, base + offset};
};
constexpr pl::identity identity{"value", "work", "v", "f32", "exact", {{"m", -2, 2}}};
constexpr pl::strings components{"dp", "column"};
constexpr pl::placement placed{"tmem", "cell", components, mapper,
    pl::coord{"m", -2, 2}, pl::bind("base")};
constexpr pl::readiness ready{"ready", "done"};
PL_EXPORT_TENSOR(value, identity, placed, pl::strings{}, ready, pl::fixed_bindings{}, pl::lends{}, pl::calls{"produce", nullptr});
PL_BINDING("base", "int", pl::coord{"base", 0, 2}, "owner", "column", "cta", "work");
struct Sixteen {
  int a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p;
};
PL_PARAMS(Sixteen, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
PL_HINT("register_budget", 128);
