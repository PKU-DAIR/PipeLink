// Keep the generated stub first: its __forceinline__ trampoline definitions must
// be visible when the interpreter role bodies are compiled.  dispatch.cu may
// precede the stage definitions because it emits extern declarations for every
// role entry before defining pipelink_kernel.
#include "binding_stub.cu"
#include "dispatch.cu"

#include "interpreter.cu"
#include "ops/noop.cu"
#include "ops/qkv.cu"
#ifdef MK_ATTENTION_CUTE
#include "ops/attn_partial_cute.cu"
#else
#include "ops/attn_partial.cu"
#endif
#include "ops/attn_reduction.cu"
#include "ops/o_proj.cu"
#include "ops/upgate.cu"
#include "ops/down_proj.cu"
#include "ops/lm_head.cu"
