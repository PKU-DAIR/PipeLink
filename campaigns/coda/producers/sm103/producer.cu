#include "mainloop.cuh"
#include <pipelink/stage.hpp>
#include <pipelink/host_view.h>
#include <pipelink_host_abi_v2.h>
extern "C" __device__ int pipelink_coda_operands();
extern "C" __device__ int pipelink_coda_state();
extern "C" __device__ pipelink_tensor_source coda_produce(const void* view, coda_tile_args* args, void* frame) {
  extern __shared__ __align__(1024) unsigned char coda_shared[];
  auto& operands=*reinterpret_cast<coda_sm103::Operands*>(coda_shared+pipelink_coda_operands());
  auto& state=*reinterpret_cast<coda_sm103::State*>(coda_shared+pipelink_coda_state());
  auto& params=*static_cast<CodaSm103Params const*>(static_cast<pipelink_host_view const*>(view)->residue);
  return pl::source_of_tmem(coda_sm103::produce(params,*args,operands,state));
}
extern "C" __device__ void coda_release(const void* view, coda_tile_args* args) {
  extern __shared__ __align__(1024) unsigned char coda_shared[];
  coda_sm103::release<false>(*reinterpret_cast<coda_sm103::State*>(coda_shared+pipelink_coda_state()));
}
PL_TYPE(coda_tile_args, a, b, M, N, K, m0, n0);
PL_STAGE("coda.producer.sm103", "1.0.0");
PL_CALL(tensor_produce, coda_produce, pl::consume, "produce");
PL_CALL(tensor_release, coda_release, pl::consume, "release");
PL_CLAIM_SMEM("pipelink_coda_operands", coda_sm103::Operands,
              pl::during{"accumulator", "tile_work", "tensor_ready"});
PL_CLAIM_SMEM("pipelink_coda_state", coda_sm103::State, "kernel");
PL_CLAIM("pipelink_coda_tmem", "tmem", 256, 256, "kernel");
PL_TOPOLOGY(pl::topology{2, pl::cluster{{2,1,1}, pl::launch{2,{2,1,1}}}, false, "static", {}});
PL_BINDING("tmem_base", "bits[32]", pl::values{"tmem_base", "bits[32]", {0,256}},
           "tensor_producer", "column", "tensor_import_site", "tensor_allocated_to_tensor_reclaimed");
PL_EXPORT_TENSOR(accumulator,
  pl::identity{"accumulator","tile_work","completed_k","f32","bf16_dot_fp32_completed",
               {{"m",0,128},{"n",0,256}}},
  pl::placement{"tmem","dp/column",{"dp","column"}, [](int m,int n,pl::bound base) {
    return std::array<int,2>{m,n+int(base)};
  }, pl::coord{"m",0,128},pl::coord{"n",0,256},pl::bind("tmem_base","bits[32]")},
  pl::strings{"tmem_packet_receipt"},pl::readiness{"tensor_ready","import_done"},pl::fixed_bindings{},
  pl::lends{
    pl::smem("producer_state",sizeof(coda_sm103::State),alignof(coda_sm103::State),
             "pipelink_coda_state","tensor_producer","tile_work","tensor_reclaimed"),
    {"producer_tmem","tmem",256,256,"cta","fixed","tmem_base",std::nullopt,
     "tensor_producer","tensor_allocated","tensor_reclaimed"}},
  pl::calls{"tensor_produce","tensor_release"});
PL_HOST_GROUP("coda.sm103", "coda.sm103.params", "coda_sm103", plk_header_v2,
              PLK_ABI_TYPE_ID(plk_header_v2), false);
PL_HOST_ABI_TYPE(100, CodaSm103Params, a, b, num_sms, stagger_ns);
PL_HOST_RESIDUE(2,"coda.sm103.residue",CodaSm103Params,100,true,pl::host::roles({}));
PL_HOST_SCALARS(1,pl::host::scalars(pl::host::roles({}))
  .field(pl::host::scalar(1,"coda.M","i64").caller("M"))
  .field(pl::host::scalar(2,"coda.N","i64").caller("N"))
  .field(pl::host::scalar(3,"coda.K","i64").caller("K"))
  .field(pl::host::scalar(4,"coda.stagger_ns","i64").caller("stagger_ns")));
PL_HOST_CHANNEL(1,uint16_t,pl::host::channel("coda.x","bf16").rank(2).read().caller("x"));
PL_HOST_CHANNEL(2,uint16_t,pl::host::channel("coda.w","bf16").rank(2).read().caller("w"));
PL_HOST_SCHEDULER(pl::host::scheduler("coda.sm103","coda.sm103.grid",pl::host::schema()));
