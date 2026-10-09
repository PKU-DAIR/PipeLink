#ifndef PIPELINK_COMM_OVERLAP_H
#define PIPELINK_COMM_OVERLAP_H
/* Public C types and provider prototypes of the comm owner (AG gather + RS epilogue). */
#include <stdint.h>
#include <pipelink_host_abi_v2.h>

#define COMM_MAX_WORLD 8

/* Peer tables hold every rank's allocation base as addressable from this rank. */
typedef struct CommAgResidue {
  uint64_t gather[COMM_MAX_WORLD];   /* bf16 [M, K]; rank r's rows [r*M/W, (r+1)*M/W) are its shard */
  uint64_t ready[COMM_MAX_WORLD];    /* u32 counter slots (gmem_counter binding is the index base) */
  uint64_t released[COMM_MAX_WORLD]; /* u32 [world]: released[q] = launches rank q has completed */
  uint32_t* generation;              /* local u32: launches this rank has completed */
  uint32_t* tickets;                 /* workspace u32 [2]: queue, done (zeroed per launch) */
  int32_t rank;
  int32_t world;
  int32_t rows_per_rank;
  int32_t K;
} CommAgResidue;

typedef struct CommRsResidue {
  uint64_t slots[COMM_MAX_WORLD];    /* f32 [2 parity][W source][M/W][N] receive slots */
  uint64_t counters[COMM_MAX_WORLD]; /* u32 [2 parity][M/W/128 * N/128] arrival counters */
  float* output;                    /* local f32 [M/W, N]; only this rank reduces its shard */
  uint64_t released[COMM_MAX_WORLD];
  uint64_t* generation;              /* local u64: launches this rank has completed */
  uint32_t* done;                    /* follow-on reduction CTA ticket, reset per step */
  int32_t rank;
  int32_t world;
  int32_t M;
  int32_t N;
} CommRsResidue;

typedef struct plk_role_view_comm_gather_comm {
  const void* residue;
  const void* scalars;
} plk_role_view_comm_gather_comm;

typedef struct plk_role_view_comm_epilogue_rs_epilogue {
  const void* residue;
  const void* scalars;
} plk_role_view_comm_epilogue_rs_epilogue;

#ifdef __cplusplus
extern "C" {
#endif
int pipelink_params_abi_comm_ag(plk_abi_report_v2* out);
int pipelink_params_query_comm_ag(const plk_common_v2* common, const void* args, plk_demands_v2* out);
int pipelink_params_build_comm_ag(const plk_common_v2* common, const void* args,
                                  const plk_resources_v2* resources, void* residue, size_t residue_cap,
                                  plk_transfer_report_v2* transfers, uint32_t transfer_capacity,
                                  plk_session** session);
int pipelink_params_release_comm_ag(plk_session* session);
int pipelink_params_abi_comm_rs(plk_abi_report_v2* out);
int pipelink_params_query_comm_rs(const plk_common_v2* common, const void* args, plk_demands_v2* out);
int pipelink_params_build_comm_rs(const plk_common_v2* common, const void* args,
                                  const plk_resources_v2* resources, void* residue, size_t residue_cap,
                                  plk_transfer_report_v2* transfers, uint32_t transfer_capacity,
                                  plk_session** session);
int pipelink_params_release_comm_rs(plk_session* session);
#ifdef __cplusplus
}
#endif
#endif
