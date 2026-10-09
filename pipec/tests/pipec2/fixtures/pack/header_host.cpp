#include <pipelink/stage.hpp>
#include <pipelink_host_abi_v2.h>

struct HostArgs { unsigned char tag; uint64_t count; };
struct HostResidue { uint64_t state; };
struct HostEmpty {};

PL_STAGE("fixture.host", "1");
PL_NO_PARAMS();
#if !defined(HOST_HOSTLESS)
PL_HOST_TARGET();
#endif

#if defined(HOST_TARGET_ONLY) || defined(HOST_HOSTLESS)
#elif defined(HOST_DECLARATIONS_ONLY)
PL_HOST_DECLARATIONS("fixture.author");
PL_HOST_FACT(pl::host::fact("fixture.width").u64(16)
    .owner("fixture.author").stage("fixture.host").consumers({}));
PL_HOST_SCHEDULER(pl::host::scheduler("fixture.integration"));
#else
#if defined(__CUDACC__)
extern "C" __device__ void fixture_host_run(const void*, unsigned) {}
#else
extern "C" void fixture_host_run(const void*, unsigned) {}
#endif
PL_ROLE(run, fixture_host_run, 1);
PL_VARIANT(0, {{"pipelink_choice", 1}});
PL_VARIANT(1, {{"pipelink_choice", 2}});
PL_HOST_VARIANTS("fixture.peer", pl::host::variants({{{"pipelink_choice", 1}}}));
#if defined(HOST_NEUTRAL_ARGS)
PL_HOST_GROUP("fixture.integration", "fixture.params", "fixture",
    plk_header_v2, PLK_ABI_TYPE_ID(plk_header_v2), true);
PL_HOST_RESIDUE(8, "fixture.residue", HostResidue, 2, true, pl::host::roles({"fixture.host:run"}));
#elif defined(HOST_ZERO_STATE)
PL_HOST_GROUP("fixture.integration", "fixture.params", "fixture", HostArgs, 1, false);
PL_HOST_RESIDUE(8, "fixture.residue", HostResidue, 2, false, pl::host::roles({"fixture.host:run"}));
#else
PL_HOST_GROUP("fixture.integration", "fixture.params", "fixture", HostArgs, 1, true);
PL_HOST_RESIDUE(8, "fixture.residue", HostResidue, 2, true, pl::host::roles({"fixture.host:run"}));
#endif
PL_HOST_ABI_TYPE(1, HostArgs, count, tag);
PL_HOST_ABI_TYPE(2, HostResidue, state);
#if !defined(HOST_EMPTY_ID)
#define HOST_EMPTY_ID 3
#endif
PL_HOST_ABI_EMPTY(HOST_EMPTY_ID, HostEmpty);
#if defined(HOST_REDECLARE_ID)
PL_HOST_ABI_TYPE(HOST_REDECLARE_ID, plk_header_v2, abi_version, struct_bytes);
#endif
PL_HOST_SCALARS(1, pl::host::scalars(pl::host::roles({"fixture.host:run"}))
    .field(pl::host::scalar(1, "fixture.M", "u64").caller("M")));
PL_HOST_CHANNEL(1, uint8_t, pl::host::channel("fixture.tensor", "u8")
    .logical("s4").rank(1).read().caller("tensor"));
PL_HOST_WORKSPACE(1, alignof(HostArgs), pl::host::workspace("fixture.workspace")
    .owner_initialized("fixture.integration"));
PL_ENDPOINT(tensor, pl::endpoint("tensor", pl::produce, "run")
    .space("smem").scope("cta").proxy("async").tile("s4", {32})
    .layout(pl::affine{{32}, {1}, 4})
    .protocol(pl::mbarrier{
        pl::side{"pipelink_full", pl::arrive{"run", 1}, 16},
        pl::side{"pipelink_empty", pl::arrive{"run", 1}, 0},
        pl::init{nullptr, 0, "stage", 0}}));
PL_HOST_TRANSFER(1, pl::host::transfer("fixture.load", "fixture.tensor")
    .device("fixture.host", "tensor").read());
#if defined(HOST_MULTIPLE_RESOURCES)
PL_HOST_DOMAIN(4, "tp");
PL_HOST_DOMAIN(2, "ep");
PL_HOST_CHANNEL(2, uint8_t, pl::host::channel("fixture.other", "u8").caller("other").peer("tp").multicast("tp"));
PL_HOST_WORKSPACE(2, alignof(HostArgs), pl::host::workspace("fixture.zero").zero().reset_launch());
PL_HOST_TRANSFER(2, pl::host::transfer("fixture.other_load", "fixture.other")
    .device("fixture.host", "tensor").read());
#endif
PL_HOST_FACT(pl::host::fact("fixture.width").u64(sizeof(HostArgs))
    .owner("fixture.integration").stage("fixture.host").consumers({}));
PL_HOST_FACT(pl::host::fact("fixture.tile").tuple_i32({16, 32})
    .owner("fixture.integration").stage("fixture.host").consumers({"fixture.peer"}));
PL_HOST_FACT(pl::host::fact("fixture.depth").i32(0)
    .variants({{{{"pipelink_choice", {1}}}, 1}, {{{"pipelink_choice", {2}}}, 2}})
    .owner("fixture.integration").stage("fixture.host").consumers({}));
PL_HOST_SCHEDULER(pl::host::scheduler("fixture.integration", "pipelink.static-grid.v1",
    pl::host::schema()
        .required("tile_extents", pl::host::leaf::tuple({
            pl::host::leaf::static_("i32"), pl::host::leaf::static_("i32")}))
        .required("grid_axes", pl::host::leaf::enumeration({"M"})))
    .input("tile_extents", pl::host::constant({16, 32}))
    .input("grid_axes", pl::host::constant("M"))
    .workspaces({1}));
#endif

#if defined(HOST_INERT)
int main() { return 0; }
#endif
