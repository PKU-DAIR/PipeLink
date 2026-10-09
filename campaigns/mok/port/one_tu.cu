// Same stage sources and link-time bindings; no alternate operator implementation.
#include MOK_BINDING_STUB
#include MOK_INTERPRETER
#include "comm/comm.cu"
#include "gemm/gemm.cu"
#include "act/act.cu"
// Generated role views deliberately restate the public provider records.
namespace mok_control_dispatch {
#include MOK_DISPATCH
}
