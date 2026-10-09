#include <pipelink/stage.hpp>
// None of these names/types exist; inert annotations cannot parse their contents.
PL_STAGE(unknown_id, unknown_version);
PL_PARAMS(UnknownParams, field);
PL_NO_PARAMS();
PL_PARAMS_OPAQUE(UnknownOpaqueParams);
PL_TYPE(UnknownType, unsigned value;);
PL_ROLE(missing_role, unknown_fn, unknown_warps);
PL_EXPORT(missing_export, unknown_fn);
PL_CALL(missing_call, unknown_fn, pl::produce, unknown_role);
PL_ENDPOINT(unknown, pl::endpoint(unknown).protocol(unknown));
PL_CLAIM(unknown_handle, unknown_domain, unknown_size, unknown_align, unknown_lifetime);
PL_CLAIM_SMEM(unknown_handle, UnknownStorage, unknown_lifetime);
PL_CLAIM_EX(pl::claim(unknown).extent(unknown));
PL_CLAIM_REGS(unknown_handle, unknown_group, {unknown_value, unknown_other}, unknown_role);
PL_VARIANT(unknown_index, {{unknown_handle, unknown_value}});
PL_VARIANT_ENTRY(unknown_index, {{unknown_handle, unknown_value}}, unknown_fn);
PL_DESCRIPTOR(UnknownDescriptor);
PL_TOPOLOGY(pl::topology{unknown_value});
PL_BINDING(unknown, unknown_sort, unknown_domain, unknown_owner, unknown_units, unknown_scope, unknown_lifetime);
PL_SITE(unknown, unknown_role, pl::all_threads, unknown_invocation, unknown_guarantees, unknown_observes);
PL_EXPORT_TENSOR(unknown, unknown_identity, unknown_placement, unknown_capabilities, unknown_readiness, unknown_fixed, unknown_backing);
PL_IMPORT_TENSOR(unknown, unknown_fn, unknown_identity, unknown_args, unknown_demand, unknown_sites, unknown_completion, unknown_lends);
PL_HINT(unknown_key, unknown_value);
int main() { return 0; }
