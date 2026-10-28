#pragma once
#include <pipelink_host_abi_v2.h>
#ifdef __cplusplus
extern "C" {
#endif
int pipelink_params_abi_coda_sm103(plk_abi_report_v2* out);
int pipelink_params_query_coda_sm103(const plk_common_v2* common, const void* args, plk_demands_v2* out);
int pipelink_params_build_coda_sm103(const plk_common_v2* common, const void* args,
    const plk_resources_v2* resources, void* residue, size_t residue_cap,
    plk_transfer_report_v2* transfers, uint32_t transfer_capacity, plk_session** session);
int pipelink_params_release_coda_sm103(plk_session* session);
#ifdef __cplusplus
}
#endif
