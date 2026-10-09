#include <cstdint>
#include <pipelink/stage.hpp>

struct Params { float* output; unsigned count; };
struct alignas(16) SharedStorage { unsigned char bytes[128]; };

extern "C" __device__ void pl_import(pipelink_tensor_source source, float* fragment);
extern "C" __device__ void pl_import_ready();
extern "C" __device__ void pl_import_done();
extern "C" __device__ float plx_scale(float value) { return value * 2.0f; }
extern "C" __device__ void plx_role_x(const void* raw, unsigned variant) {
    const auto* params = static_cast<const Params*>(raw);
    float fragment[2];
    pl_import_ready();
    pl_import(pl::source_of(params->output), fragment);
    pl_import_done();
    if (params->count > variant) params->output[variant] = plx_scale(fragment[0]);
}
extern "C" __device__ void test_pack_min_variant_1(const void* raw, unsigned variant) {
    plx_role_x(raw, variant);
}

PL_STAGE("test.pack-min", "1.0.0");
PL_PARAMS(Params, output, count);
PL_ROLE(x, plx_role_x, 1);
PL_EXPORT(scale, plx_scale);
PL_CLAIM_SMEM("pipelink_test_pack_min_smem", SharedStorage, "kernel", "x");
PL_CLAIM_EX(pl::claim("pipelink_min_named", "named_barrier").extent(1).align(1)
            .lifetime("phase:compute").at(3).alias("MinNamed").roles({"x"}));
PL_ENDPOINT(sync, pl::endpoint("sync", pl::produce, "x")
    .space("smem").scope("cta").proxy("generic").tile("f32", {4})
    .layout(pl::affine{{4}, {1}, 32})
    .protocol(pl::named_barrier{"pipelink_min_named", 32})
    .bind("base", "pipelink_test_pack_min_smem"));
PL_VARIANT(0, {{"pipelink_min_depth", 2}});
PL_VARIANT_ENTRY(1, {{"pipelink_min_depth", 3}}, test_pack_min_variant_1);
PL_DESCRIPTOR(Params);
PL_SITE(import_site, "x", pl::all_threads, "once",
        pl::guarantees{"enter_once", "converged", "full_warps", "no_early_return"},
        pl::strings{"ready"});
PL_IMPORT_TENSOR(input, pl_import,
    pl::identity{"accumulator", "work", "v1", "f32", "exact", {{"m", 0, 32}, {"n", 0, 2}}},
    pl::args{{"source", "source_base"}, {"fragment", "dest_fragment"}},
    pl::demand_positions{pl::coord{"t", 0, 32}, pl::coord{"p", 0, 2},
                        [](int t, int p) { return std::array<int, 4>{t, p, t, p}; }},
    pl::strings{"import_site"}, "done",
    pl::lends{{"fragment", "reg", 2, 1, "thread", "fixed", nullptr, 0, "x", "ready", "done"},
              pl::cta_barrier("outer", 0, "x", "ready", "done", pl::sync{2,1})},
    {pl::rendezvous{"ready", "outer", pl::all_threads}, pl::rendezvous{"done", "outer", pl::all_threads}});
