// Public caller args for the interpreter's four-export host ABI v2 provider.
// Device resources are owner-typed args; build writes the opaque ThunderKittens
// globals into the published residue. Native clients need no ThunderKittens types.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <pipelink_host_abi_v2.h>

#define MK_PROVIDER_ARGS_VERSION 1u
#define MK_GLOBALS_BYTES 9600u
#define MK_GLOBALS_ALIGN 128u

#define MK_HOST_ROLE_TYPES(X) \
  X(30, plk_role_view_org_pipelink_megakernel_interpreter_consumer) \
  X(31, plk_role_view_org_pipelink_megakernel_interpreter_loader) \
  X(32, plk_role_view_org_pipelink_megakernel_interpreter_storer) \
  X(33, plk_role_view_org_pipelink_megakernel_interpreter_launcher) \
  X(34, plk_role_view_org_pipelink_megakernel_interpreter_controller)

#ifdef __cplusplus
namespace mk_host {
struct InterpreterView {
  const void* residue;
  const void* scalars;
};
#define MK_DECLARE_ROLE_VIEW(id, T) using T = InterpreterView;
MK_HOST_ROLE_TYPES(MK_DECLARE_ROLE_VIEW)
#undef MK_DECLARE_ROLE_VIEW
}  // namespace mk_host
#endif

// Every pointer is a CUDA device address (CUdeviceptr). Dimension fields fill the -1 axes of the
// corresponding ThunderKittens `gl` types in llama.cuh (llama_1b_globals); fixed axes are
// hard-coded in the provider. Layouts (batch, depth, rows, cols):
//   Bar          uint  (1, num_layers, bar_rows=10, 48)
//   instructions int   (1, num_sms, num_instructions, 32)
//   timings      int   (1, num_sms, num_instructions, 128)
//   qkv_weights  bf16  (1, num_layers, 3072, 2048)     o_weights   (1, num_layers, 2048, 2048)
//   up/gate      bf16  (1, num_layers, 8192, 2048)     down        (1, num_layers, 2048, 8192)
//   attn/mlp norm bf16 (1, 1, num_layers, 2048)        lm_head_norm (1, 1, 1, 2048)
//   lm_head_weights bf16 (1, 1, vocab_size, 2048)
//   k_cache/v_cache bf16 (num_layers, max_seq_len, 8, 64)
//   rope_cos/sin f32   (1, 1, rope_rows, 64)
//   hidden_states, q_post_rope, attn_out bf16 (1,1,1,2048); silu_out (1,1,1,8192); logits (1,1,1,vocab_size)
//   attn_lse_intermediates f32 (1, 1, 32, num_sms)    attn_out_intermediates f32 (1, 32, num_sms, 64)
typedef struct MkProviderArgs {
  uint32_t version;  // MK_PROVIDER_ARGS_VERSION
  uint32_t reserved0;
  uint64_t bar;
  uint64_t instructions;
  uint64_t timings;
  uint64_t qkv_weights;
  uint64_t attn_norm_weights;
  uint64_t o_weights;
  uint64_t mlp_norm_weights;
  uint64_t up_weights;
  uint64_t gate_weights;
  uint64_t down_weights;
  uint64_t lm_head_norm_weights;
  uint64_t lm_head_weights;
  uint64_t k_cache;
  uint64_t v_cache;
  uint64_t rope_cos;
  uint64_t rope_sin;
  uint64_t hidden_states;
  uint64_t q_post_rope;
  uint64_t attn_out;
  uint64_t attn_lse_intermediates;
  uint64_t attn_out_intermediates;
  uint64_t silu_out;
  uint64_t logits;
  int32_t num_sms;           // grid size; must equal the device SM count (MK uses %smid)
  int32_t num_instructions;  // instruction rows per SM
  int32_t max_seq_len;       // KV cache rows
  int32_t rope_rows;
  int32_t num_layers;        // 16
  int32_t bar_rows;          // 10
  int32_t vocab_size;        // 128256
  uint32_t pos_id;
  float attn_scale;
  float rms_norm_eps;
  int32_t skip_attn_reduction;  // bool
  int32_t reserved1;
} MkProviderArgs;
