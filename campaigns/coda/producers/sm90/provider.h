#pragma once
#include <pipelink_host_abi_v2.h>
#ifdef __cplusplus
extern "C" {
#endif
int pipelink_params_abi_coda_sm90(plk_abi_report_v2* out);
int pipelink_params_query_coda_sm90(const plk_common_v2*, const void*, plk_demands_v2*);
int pipelink_params_build_coda_sm90(const plk_common_v2*, const void*, const plk_resources_v2*,
                                 void*, size_t, plk_transfer_report_v2*, uint32_t, plk_session**);
int pipelink_params_release_coda_sm90(plk_session*);
#ifdef __cplusplus
}
#endif
