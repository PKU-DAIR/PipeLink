#include <pipelink/stage.hpp>
PL_SITE(nobody, "worker", pl::participants{pl::coord{"t",0,32}, [](int) { return false; }}, "work", pl::guarantees{}, pl::strings{});
