Generated from the profile's authoring ledger; do not edit.

# CODA authoring ledger

Profile: `sm120`. The ledger is a source census of the hand-CUDA consumer and a transcription of the TileLang frontend's own ledger.

## Counting policy

The ledger counts the contract a consumer author writes to compose with a packaged producer. One record is one listed semantic field; an aggregate record keeps its grouped fields together, so the count is neither a line count nor a time-to-author measure. The hand-CUDA consumer is counted by a manual source census: identity scalars, the four demand equations, each guarantee, each resource and rendezvous, and each named stage or host entity, every one with its source and value. The TileLang consumer is counted exactly as the frontend's own ledger classifies its records, without reclassifying or expanding them.

Both consumers use the same owner-authored host annotation header. The hand census expands it into 11 shared records plus 1 profile-selected scheduler record; the TileLang ledger collapses it into one declared `host` record. Because the two censuses count at different granularity, the ledger claims no ratio or percentage reduction between them. Annotation counts are active `PL_*` invocations after profile conditionals and macro-wrapper expansion.

Excluded amortized work: frontend capture/lowering/checking implementation; linker/planner/model implementation; stock host ABI builtin type inventory; packaging/reflection boilerplate.
Excluded owner kernel code: SwiGLU arithmetic; output layout/swizzle and vector drain; tile-origin computation in producers; ordinary buffer/array declarations and shape assumptions.

## Counts by class

Contract-derived records follow from a contract the consumer selects; compiler-reflected records are read back from compiled types; convention records are fixed defaults. The record tables keep the raw class labels: `captured` is frontend-captured, `contract` and `contract_derived` are contract-derived, `reflected` is compiler-reflected.

| Consumer | Owner `PL_*` invocations | Declared | Frontend-captured | Contract-derived | Compiler-reflected | Convention | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| hand | 19 | 61 | 0 | 3 | 9 | 2 | 75 |
| TileLang | 9 | 7 | 14 | 19 | 20 | 1 | 61 |

TileLang's 7 declared records split into 5 explicit recipe records and 2 frontend defaults or generated records; all hand declared records are author-stated.

## Shared host annotations

Source: `campaigns/coda/consumer/host_annotations.cuh:8-31`. Both consumers use 9 active invocations: `PL_HOST_GROUP` × 1, `PL_HOST_ABI_TYPE` × 2, `PL_HOST_RESIDUE` × 1, `PL_HOST_SCALARS` × 1, `PL_HOST_CHANNEL` × 3, `PL_HOST_SCHEDULER` × 1.

## Hand-CUDA consumer

Annotation invocations: shared stage 10, shared host 8, profile selected host scheduler 1 (total 19). Stage macros: `PL_TYPE` × 1, `PL_STAGE` × 1, `PL_ROLE` × 1, `PL_TOPOLOGY` × 1, `PL_CALL` × 2, `PL_BINDING` × 1, `PL_SITE` × 1, `PL_CLAIM_SMEM` × 1, `PL_IMPORT_TENSOR` × 1.

Author-stated records by partition: shared stage 41, profile selected lends 1, profile geometry 7, shared host 11, profile host 1.

The consumer states a fixed demand and requests no frontend specialization. Resolved demand: m = `t / 2`, n = `32*chunk + 16*(t % 2) + p`, participant = `t + 0`, position = `p` over t ∈ [0, 256), p ∈ [0, 16), chunk ∈ [0, 8). Selected link route: `reg>store|rendezvous|load>reg`.

The owner chunk schedule (`campaigns/coda/consumer/epilogue.cu:54-81`: 8 chunks, 1 generation, per chunk completion) is kernel code and is not counted.

| Field | Class | Partition | Source | Value |
|---|---|---|---|---|
| `declarations.argument_type` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:6` | `{"type":"coda_tile_args","members":["a","b","M","N","K","m0","n0"]}` |
| `declarations.stage_id` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:7` | `"coda.swiglu"` |
| `declarations.version` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:7` | `"1.0.0"` |
| `declarations.params` | contract_derived | shared_stage | `implied by PL_HOST_GROUP (campaigns/coda/consumer/host_annotations.cuh)` | `"no raw parameter blob; host view"` |
| `declarations.role` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:8` | `{"name":"epilogue","entry":"plx_role_coda_epilogue","warps_expression":"CODA_THREADS / 32"}` |
| `topology.shared_policy` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:9-11` | `{"cluster_yz":[1,1],"launch_yz_max":[1,1],"persistent":false,"scheduler":"static","multicast":[]}` |
| `calls.produce` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:12` | `{"endpoint":"tensor_produce","symbol":"pipelink_call_tensor_produce","direction":"produce","role":"epilogue","export":"produce"}` |
| `calls.release` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:13` | `{"endpoint":"tensor_release","symbol":"pipelink_call_tensor_release","direction":"produce","role":"epilogue","export":"release"}` |
| `bindings.chunk` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:14-15` | `{"name":"chunk","sort":"bits[32]","domain":[0,8],"owner":"epilogue","units":"chunk","observation_scope":"tensor_import_site","lifetime":"tensor_ready_to_import_done"}` |
| `sites.identity` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:16-19` | `{"name":"tensor_import_site","role":"epilogue","invocation":"tile_work"}` |
| `sites.participants` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:17` | `{"domain":"t in [CODA_FIRST_DATA_THREAD, CODA_FIRST_DATA_THREAD+256)","map":"t"}` |
| `sites.guarantees.enter_once` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:18` | `true` |
| `sites.guarantees.converged` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:18` | `true` |
| `sites.guarantees.full_warps` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:18` | `true` |
| `sites.guarantees.no_early_return` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:18` | `true` |
| `sites.observes` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:19` | `["tensor_ready","chunk"]` |
| `claims.output` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:20-21` | `{"handle":"pipelink_coda_output_smem","storage_type":"coda_epilogue::OutputTile","lifetime":{"tensor":"acc","from":"tensor_ready","to":"kernel_end"},"owner":"epilogue"}` |
| `identity.result` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:23-24` | `"accumulator"` |
| `identity.work` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:23-24` | `"tile_work"` |
| `identity.version` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:23-24` | `"completed_k"` |
| `identity.dtype` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:23-24` | `"f32"` |
| `identity.interpretation` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:23-24` | `"bf16_dot_fp32_completed"` |
| `identity.coords.m.name` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:24` | `"m"` |
| `identity.coords.m.lo` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:24` | `0` |
| `identity.coords.m.hi` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:24` | `128` |
| `identity.coords.m.sort` | convention | shared_stage | `campaigns/coda/consumer/epilogue.cu:24 (pl::coord default)` | `"int"` |
| `identity.coords.n.name` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:24` | `"n"` |
| `identity.coords.n.lo` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:24` | `0` |
| `identity.coords.n.hi` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:24` | `256` |
| `identity.coords.n.sort` | convention | shared_stage | `campaigns/coda/consumer/epilogue.cu:24 (pl::coord default)` | `"int"` |
| `import.argument_meanings` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:25` | `{"source":"source_base","chunk":"binding","dest":"dest_fragment","scratch":"scratch"}` |
| `demand.domain` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:26` | `{"t":[0,256],"p":[0,16],"chunk_binding":"bits[32]"}` |
| `demand.equations.m` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:27-29` | `"row(t)"` |
| `demand.equations.n` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:27-29` | `"32*chunk + 16*half_row(t) + p"` |
| `demand.equations.participant` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:27-29` | `"t + CODA_FIRST_DATA_THREAD"` |
| `demand.equations.position` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:27-29` | `"p"` |
| `import.sites` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:30` | `["tensor_import_site"]` |
| `import.completion_event` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:30` | `"import_done"` |
| `lends.frame` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:32` | `{"kind":"frame","extent":128,"handle":"frame","owner":"epilogue","from":"tile_work","to":"import_done"}` |
| `lends.scratch` | declared | profile_selected_lends | `campaigns/coda/consumer/epilogue.cu:34` | `{"kind":"smem","extent_expression":"CODA_IMPORT_SCRATCH_BYTES","align":128,"handle":"pipelink_coda_acc_smem","owner":"epilogue","from":"tensor_ready","to":"import_done"}` |
| `lends.dest` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:36` | `{"kind":"regs","extent":16,"handle":"dest","owner":"epilogue","from":"tensor_ready","to":"import_done"}` |
| `lends.metadata` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:37` | `{"kind":"metadata","extent":2,"handle":"acc_metadata","owner":"epilogue","from":"tensor_ready","to":"import_done"}` |
| `lends.outer` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:41` | `{"kind":"cta_barrier","base":0,"owner":"epilogue","from":"tile_work","to":"import_done","sync":[2,1]}` |
| `rendezvous.tensor_ready` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:42` | `{"barrier":"acc_outer","cohort":"all_threads"}` |
| `rendezvous.import_done` | declared | shared_stage | `campaigns/coda/consumer/epilogue.cu:43` | `{"barrier":"acc_outer","cohort":"all_threads"}` |
| `geometry.CODA_THREADS` | declared | profile_geometry | `campaigns/coda/consumer/geometry.cuh:20-24` | `256` |
| `geometry.CODA_FIRST_DATA_THREAD` | declared | profile_geometry | `campaigns/coda/consumer/geometry.cuh:20-24` | `0` |
| `geometry.CODA_CTA_GROUP` | declared | profile_geometry | `campaigns/coda/consumer/geometry.cuh:20-24` | `1` |
| `geometry.CODA_CLUSTER_X` | declared | profile_geometry | `campaigns/coda/consumer/geometry.cuh:20-24` | `1` |
| `geometry.CODA_IMPORT_SCRATCH_BYTES` | declared | profile_geometry | `campaigns/coda/consumer/geometry.cuh:20-24` | `16384` |
| `geometry.row` | declared | profile_geometry | `campaigns/coda/consumer/geometry.cuh:34-40` | `"t / 2"` |
| `geometry.half_row` | declared | profile_geometry | `campaigns/coda/consumer/geometry.cuh:41-47` | `"t % 2"` |
| `host.group` | declared | shared_host | `campaigns/coda/consumer/host_annotations.cuh:8-9` | `{"owner":"coda.epilogue","group":"coda.epilogue.params","provider_suffix":"coda_epilogue","args_type":"plk_header_v2","allocated":false}` |
| `host.abi.residue_members` | declared | shared_host | `campaigns/coda/consumer/host_annotations.cuh:10` | `{"type_id":200,"type":"CodaEpilogueParams","members":["a","b","output","M","N","K"]}` |
| `host.abi.role_view_members` | declared | shared_host | `campaigns/coda/consumer/host_annotations.cuh:11-13` | `{"type_id":201,"type":"plk_role_view_coda_swiglu_epilogue","members":["residue","scalars"]}` |
| `host.residue` | declared | shared_host | `campaigns/coda/consumer/host_annotations.cuh:14-15` | `{"field_id":2,"name":"coda.epilogue.residue","type_id":200,"required":true,"readers":["coda.swiglu:epilogue"]}` |
| `host.scalar_group` | declared | shared_host | `campaigns/coda/consumer/host_annotations.cuh:16` | `{"id":1,"readers":["coda.swiglu:epilogue"]}` |
| `host.scalars.M` | declared | shared_host | `campaigns/coda/consumer/host_annotations.cuh:17` | `{"id":1,"name":"coda.M","type":"i64","caller":"M"}` |
| `host.scalars.N` | declared | shared_host | `campaigns/coda/consumer/host_annotations.cuh:18` | `{"id":2,"name":"coda.N","type":"i64","caller":"N"}` |
| `host.scalars.K` | declared | shared_host | `campaigns/coda/consumer/host_annotations.cuh:19` | `{"id":3,"name":"coda.K","type":"i64","caller":"K"}` |
| `host.channels.x` | declared | shared_host | `campaigns/coda/consumer/host_annotations.cuh:20` | `{"id":1,"element_type":"uint16_t","name":"coda.x","dtype":"bf16","rank":2,"access":"read","caller":"x"}` |
| `host.channels.w` | declared | shared_host | `campaigns/coda/consumer/host_annotations.cuh:21` | `{"id":2,"element_type":"uint16_t","name":"coda.w","dtype":"bf16","rank":2,"access":"read","caller":"w"}` |
| `host.channels.out` | declared | shared_host | `campaigns/coda/consumer/host_annotations.cuh:22` | `{"id":3,"element_type":"uint16_t","name":"coda.out","dtype":"bf16","rank":2,"access":"write","caller":"out"}` |
| `host.scheduler` | declared | profile_host | `campaigns/coda/consumer/host_annotations.cuh:23-30` | `"coda.sm120.owner"` |
| `host.target_publication` | contract_derived | shared_host | `implied by PL_HOST_GROUP (campaigns/coda/consumer/host_annotations.cuh)` | `"host {target} dtype section"` |
| `compiler.argument_layout` | reflected | compiler | `campaigns/coda/out/sm120/packed/hand.plink#/types/0` | `{"align":8,"bytes":40,"members":[{"align":8,"bytes":8,"ctype":"const void*","extent":0,"name":"a","offset":0},{"align":8,"bytes":8,"ctype":"const void*","extent":0,"name":"b","offset":8},{"align":4,"bytes":4,"ctype":"int32_t","extent":0,"name":"M","offset":16},{"align":4,"bytes":4,"ctype":"int32_t","extent":0,"name":"N","offset":20},{"align":4,"bytes":4,"ctype":"int32_t","extent":0,"name":"K","offset":24},{"align":4,"bytes":4,"ctype":"int32_t","extent":0,"name":"m0","offset":28},{"align":4,"bytes":4,"ctype":"int32_t","extent":0,"name":"n0","offset":32}],"name":"coda_tile_args"}` |
| `compiler.calls.produce` | reflected | compiler | `campaigns/coda/consumer/epilogue_abi.cuh:5` | `"pipelink_tensor_source(const void*, coda_tile_args*, void*)"` |
| `compiler.calls.release` | reflected | compiler | `campaigns/coda/consumer/epilogue_abi.cuh:6` | `"void(const void*, coda_tile_args*)"` |
| `compiler.import_parameter_types` | reflected | compiler | `campaigns/coda/consumer/epilogue_abi.cuh:7` | `["pipelink_tensor_source","uint32_t","float*","void*"]` |
| `compiler.output_storage_bytes` | reflected | compiler | `campaigns/coda/consumer/output.cuh:6 (sizeof)` | `32768` |
| `compiler.output_storage_alignment` | reflected | compiler | `campaigns/coda/consumer/output.cuh:6 (alignof)` | `128` |
| `compiler.host_abi_layout.200` | reflected | compiler | `campaigns/coda/out/sm120/packed/hand.plink#/host/provider/abi_types (id=200)` | `{"bytes":40,"align":8}` |
| `compiler.host_abi_layout.201` | reflected | compiler | `campaigns/coda/out/sm120/packed/hand.plink#/host/provider/abi_types (id=201)` | `{"bytes":16,"align":8}` |
| `compiler.role_warps` | reflected | compiler | `campaigns/coda/consumer/epilogue.cu:8 (constant expression)` | `8` |
| `host.required_profile_contracts` | contract | profile_contract | `campaigns/coda/consumer/host_annotations.cuh:8-9 (contract selected by the host group)` | `["dtypes"]` |

## TileLang consumer

Explicit recipe records: `host`, `declarations.stage_id`, `declarations.version`, `tensor.imports.pl_import_accumulator.outputs`, `tensor.imports.pl_import_accumulator.chunk`.

Frontend defaults or generated records counted as declared:

- `declarations.launch_register_cap`: contract_stage default 255; recipe does not pass an override.
- `claims.pipelink_tilelang_warp_swiglu`: Generated whole-CTA warp claim from captured kernel geometry and fixed warp-zero convention, not a separate recipe annotation.

Further author choices in the recipe, listed here and not part of the counts:

| Choice | Value | Source |
|---|---|---|
| partner/export selection | `{"package_input":"PIPELINK_ACC_PARTNER","export":"accumulator"}` | `campaigns/coda/consumer-tilelang/swiglu.py:8-9` |
| specialization requested | `true` | `campaigns/coda/consumer-tilelang/swiglu.py:21` |
| scratch capacity policy | `0` | `campaigns/coda/consumer-tilelang/swiglu.py:9` |
| completion policy | `"final"` | `campaigns/coda/consumer-tilelang/swiglu.py:24` |

The topology has no record of its own (`campaigns/coda/out/sm120/packed/tilelang.plink#/topology`): `{"cluster":{"compiled":[1,1,1],"launch":{"max":[1,1,1],"x_multiple_of":1}},"cta_group":{"fixed":1},"multicast":[],"persistent":false,"scheduler":"static"}`. Producer omits topology; the emitted consumer uses the default topology. This default is not a separate raw frontend ledger entry and is not added to its convention count.

### Specialization

1 of 1 requested imports specialized; 0 fell back.

| Import | Export | Adopted | Logical chunk | Fragment | Captured thread | Captured position | Owners per chunk | Chunk thread offsets | Barrier guard splits | Completion | Route |
|---|---|---|---|---|---|---|---|---|---:|---|---|
| `pl_import_accumulator` | accumulator | True | 128×32 | 64 | `_i // 64 * 32 + _i % 8 * 4 + _j % 8 // 2` | `_j // 8 * 16 + _i % 64 // 8 * 2 + _j % 2` | 64 from `64*(chunk//2)` | 0, 0, 64, 64, 128, 128, 192, 192 | 1 | final | `reg>copy>reg` |

Why `pl_import_accumulator` is specialized: The expressible mma.sync register map is retained with exact per-chunk whole-warp cohort translations; each chunk has 64 positions on 64 active owners.

### Records

| Field | Class | Source |
|---|---|---|
| `host` | declared | `STAGE.host owner annotations` |
| `declarations.stage_id` | declared | `STAGE.stage_id` |
| `declarations.version` | declared | `STAGE.version` |
| `declarations.params` | captured | `PrimFunc signature/kernel` |
| `declarations.launch_register_cap` | declared | `STAGE.launch_register_cap` |
| `declarations.roles` | captured | `PrimFunc signature/kernel` |
| `params.bytes` | reflected | `sizeof(Params)` |
| `params.align` | reflected | `alignof(Params)` |
| `params.fields.a.offset` | reflected | `offsetof(Params, a)` |
| `params.fields.a.bytes` | reflected | `sizeof(((Params*)nullptr)->a)` |
| `params.fields.a.align` | reflected | `__alignof__(((Params*)nullptr)->a)` |
| `params.fields.b.offset` | reflected | `offsetof(Params, b)` |
| `params.fields.b.bytes` | reflected | `sizeof(((Params*)nullptr)->b)` |
| `params.fields.b.align` | reflected | `__alignof__(((Params*)nullptr)->b)` |
| `params.fields.output.offset` | reflected | `offsetof(Params, output)` |
| `params.fields.output.bytes` | reflected | `sizeof(((Params*)nullptr)->output)` |
| `params.fields.output.align` | reflected | `__alignof__(((Params*)nullptr)->output)` |
| `params.fields.M.offset` | reflected | `offsetof(Params, M)` |
| `params.fields.M.bytes` | reflected | `sizeof(((Params*)nullptr)->M)` |
| `params.fields.M.align` | reflected | `__alignof__(((Params*)nullptr)->M)` |
| `params.fields.N.offset` | reflected | `offsetof(Params, N)` |
| `params.fields.N.bytes` | reflected | `sizeof(((Params*)nullptr)->N)` |
| `params.fields.N.align` | reflected | `__alignof__(((Params*)nullptr)->N)` |
| `params.fields.K.offset` | reflected | `offsetof(Params, K)` |
| `params.fields.K.bytes` | reflected | `sizeof(((Params*)nullptr)->K)` |
| `params.fields.K.align` | reflected | `__alignof__(((Params*)nullptr)->K)` |
| `claims.pipelink_tilelang_warp_swiglu` | declared | `STAGE.roles.swiglu.warp_range + COMPOSITION.forced_bindings` |
| `tensor.imports.pl_import_accumulator.identity.coords.0.hi` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/identity/coords/0/hi` |
| `tensor.imports.pl_import_accumulator.identity.coords.0.lo` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/identity/coords/0/lo` |
| `tensor.imports.pl_import_accumulator.identity.coords.0.name` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/identity/coords/0/name` |
| `tensor.imports.pl_import_accumulator.identity.coords.0.sort` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/identity/coords/0/sort` |
| `tensor.imports.pl_import_accumulator.identity.coords.1.hi` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/identity/coords/1/hi` |
| `tensor.imports.pl_import_accumulator.identity.coords.1.lo` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/identity/coords/1/lo` |
| `tensor.imports.pl_import_accumulator.identity.coords.1.name` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/identity/coords/1/name` |
| `tensor.imports.pl_import_accumulator.identity.coords.1.sort` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/identity/coords/1/sort` |
| `tensor.imports.pl_import_accumulator.identity.dtype` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/identity/dtype` |
| `tensor.imports.pl_import_accumulator.identity.interpretation` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/identity/interpretation` |
| `tensor.imports.pl_import_accumulator.identity.result` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/identity/result` |
| `tensor.imports.pl_import_accumulator.identity.version` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/identity/version` |
| `tensor.imports.pl_import_accumulator.identity.work` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/identity/work` |
| `tensor.imports.pl_import_accumulator.calls.produce` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/calls/produce` |
| `tensor.imports.pl_import_accumulator.calls.release` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/calls/release` |
| `tensor.imports.pl_import_accumulator.readiness.after` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/readiness/after` |
| `tensor.imports.pl_import_accumulator.readiness.reuse_after` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/readiness/reuse_after` |
| `tensor.imports.pl_import_accumulator.demand` | captured | `captures.layouts.frag` |
| `tensor.imports.pl_import_accumulator.demand.site` | captured | `frontend generated import` |
| `tensor.imports.pl_import_accumulator.lends` | captured | `frontend generated import` |
| `tensor.imports.pl_import_accumulator.rendezvous` | captured | `frontend generated import` |
| `tensor.imports.pl_import_accumulator.completion` | captured | `frontend generated import` |
| `tensor.imports.pl_import_accumulator.arguments` | convention | `published member name/type binding` |
| `tensor.imports.pl_import_accumulator.outputs` | declared | `import_tensor.outputs` |
| `tensor.imports.pl_import_accumulator.chunk` | declared | `fragment.chunk axis/width` |
| `tensor.imports.pl_import_accumulator.specialization` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/placements` |
| `tensor.sites.tensor_import_site.guarantees.enter_once` | captured | `captures.pre_codegen import schedule` |
| `tensor.sites.tensor_import_site.guarantees.single_completion` | captured | `captures.pre_codegen import schedule` |
| `tensor.sites.tensor_import_site.guarantees.converged` | captured | `captures.pre_codegen import schedule` |
| `tensor.sites.tensor_import_site.guarantees.full_warps` | captured | `captures.pre_codegen import schedule` |
| `tensor.sites.tensor_import_site.guarantees.no_early_return` | captured | `captures.pre_codegen import schedule` |
| `claims.pipelink_pl_import_accumulator_staging_0.extent` | captured | `captures.physical_allocations.staging` |
| `claims.pipelink_pl_import_accumulator_staging_0.lifetime` | contract | `campaigns/coda/out/sm120/packed/producer.plink#/tensor/exports/0/readiness` |
| `captures.None` | captured | `facts.json#/variants/None` |
