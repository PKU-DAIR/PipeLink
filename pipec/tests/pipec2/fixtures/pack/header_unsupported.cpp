#include <pipelink/stage.hpp>
struct Unsupported {};
extern "C" void plx_bad(int, Unsupported);
PL_EXPORT(bad, plx_bad);
