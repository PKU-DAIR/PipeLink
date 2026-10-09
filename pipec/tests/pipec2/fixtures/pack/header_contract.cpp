#include <pipelink/stage.hpp>
#include <cstdint>
#if defined(__CUDACC__)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define DEVICE __device__
#else
#define DEVICE
#endif

struct Params {
  const void* a; const void* b; const float* bias; float* output;
  int M; int N; int K; float* row_sum;
};
struct alignas(128) SharedStorage { alignas(16) std::uint64_t barrier; char bytes[128]; };
extern "C" DEVICE void plx_role(const void*, unsigned);
extern "C" DEVICE void plx_produce(const void*);
extern "C" DEVICE void plx_release(const void*);
extern "C" DEVICE void plx_import(pipelink_tensor_source, float*, void*);
extern "C" DEVICE void plx_view(float*);
extern "C" DEVICE double plx_types(bool, char, signed char, unsigned char, short, unsigned short,
    int, unsigned, long, unsigned long, long long, unsigned long long, float, double,
    void*, const void*, const float*, double*, int* const*, const std::uint64_t*);
extern "C" DEVICE void plx_fixed(std::int8_t, std::uint8_t, std::int16_t, std::uint16_t,
    std::int32_t, std::uint32_t, std::int64_t, std::uint64_t);
#if defined(__CUDACC__)
extern "C" DEVICE __half plx_cuda(__half, __nv_bfloat16, const __half*, __nv_bfloat16*);
PL_EXPORT(cuda, plx_cuda);
PL_ROLE(cuda_role, plx_cuda, 1);
PL_CALL(cuda_call, plx_cuda, pl::consume, "cuda");
#endif

PL_STAGE("header.contract", "1.0.0");
PL_PARAMS(Params, a, b, bias, output, M, N, K, row_sum);
PL_ROLE(worker, plx_role, 4);
PL_EXPORT(types, plx_types);
PL_ROLE(types_role, plx_types, 1);
PL_CALL(types_call, plx_types, pl::produce, "types_role");
PL_EXPORT(fixed, plx_fixed);
PL_ROLE(fixed_role, plx_fixed, 1);
PL_CALL(fixed_call, plx_fixed, pl::consume, "fixed");
PL_CALL(produce, plx_produce, pl::produce, "worker", "producer_entry");
PL_CALL(release, plx_release, pl::consume, "release");
PL_CLAIM_SMEM("producer_storage", SharedStorage, "kernel", "worker");
PL_CLAIM("private_registers", "reg", 128, 1, "phase:work", "worker");
PL_TOPOLOGY(pl::topology{2, pl::cluster{{2,1,1}, pl::launch{2,{4,1,1}}}, true, "persistent", {"a"}});
PL_HINT("message", "quote\" slash\\ newline\n tab\t \x01 UTF-8: \xc3\xa9");
PL_SITE(all, "worker", pl::all_threads, "work", pl::guarantees{"enter_once","converged","full_warps","no_early_return"}, pl::strings{"ready"});
PL_SITE(even, "worker", pl::participants{pl::coord{"i",0,64}, [](int i) { return 2*i; }}, "work", pl::guarantees{"converged"}, pl::strings{"ready"});
PL_SITE(predicate, "worker", [](int t) { return t < 64; }, "work", pl::guarantees{}, pl::strings{});
PL_IMPORT_TENSOR(input, plx_import,
  pl::identity{"accumulator","work","completed_k","f32","fp16_dot_fp32_completed",{{"m",0,128},{"n",0,128}}},
  pl::args{{"partner_base","source_base"},{"fragment128","dest_fragment"},{"scratch","scratch"}},
  pl::demand_positions{pl::coord{"t",0,128},pl::coord{"p",0,128},[](int t,int p){return std::array<int,4>{t,p,t,p};}},
  pl::strings{"all"}, "import_done",
  pl::lends{pl::smem("scratch",65536,128,"pipelink_scratch","consumer","ready","import_done"),
    pl::regs("fragment",128,"fragment128","consumer","ready","import_done"),
    pl::cta_barrier("rendezvous",0,"consumer","work","import_done",pl::sync{2,1})},
  {pl::rendezvous{"ready","rendezvous",pl::all_threads}, pl::rendezvous{"import_done","rendezvous",pl::all_threads}});
PL_IMPORT_TENSOR(view, plx_view,
  pl::identity{"accumulator","work","completed_k","f32","fp16_dot_fp32_completed",{{"m",0,2}}},
  pl::args{{"view","dest_view"}},
  pl::demand_view{pl::values{"m","int",{1,0}},pl::bind("tmem_base"),"smem","byte",[](int m,pl::bound base){return std::array<int,2>{m,4*m+base};}},
  pl::strings{"all"}, "import_done", pl::lends{}, {});
PL_EXPORT_TENSOR(accumulator,
  pl::identity{"accumulator","work","completed_k","f32","fp16_dot_fp32_completed",{{"m",0,128},{"n",0,128}}},
  pl::placement{"tmem","cell",pl::strings{"dp","column"},
    [](int m,int n,pl::bound base){return std::array<int,2>{m,n+base};},
    pl::coord{"m",0,128},pl::coord{"n",0,128},pl::bind("tmem_base")},
  pl::strings{"tmem_packet_receipt"}, pl::readiness{"ready","import_done"},pl::fixed_bindings{{"partner_fixed",7}},
  pl::lends{pl::cta_barrier("owner_barrier",1,"producer","work","import_done",pl::sync{2,1}),
    pl::metadata("metadata",2,"metadata_arg","producer","work","import_done")}, pl::calls{"release", nullptr});
// Deliberately after the map: registration order cannot affect binding resolution.
#ifndef OMIT_BINDING
PL_BINDING("tmem_base","bits[32]",pl::values{"tmem_base","bits[32]",{0,128,256,384}},"producer","column","cta","work");
#endif
