#include "../../producers/sm103/mainloop.cuh"

namespace coda_sm103 {
struct NativeArgs {
  CodaSm103Params producer;
  void* output;
  int32_t M,N,K;
};
struct GemmNativeArgs {
  NativeArgs main;
  alignas(128) CUtensorMap d;
};
__device__ inline void receipt(uint32_t address, float (&v)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
    : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),"=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]),
      "=f"(v[8]),"=f"(v[9]),"=f"(v[10]),"=f"(v[11]),"=f"(v[12]),"=f"(v[13]),"=f"(v[14]),"=f"(v[15])
    : "r"(address));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
template<int Threads>
__device__ inline void epilogue(uint32_t base,coda_tile_args const& args,void* output,bf16* stage) {
  constexpr int Width=128;
  int row=threadIdx.x%128;
  #pragma unroll
  for(int chunk=0;chunk<256/(Threads/128*16);++chunk) {
    int col=chunk*(Threads/128*16)+(threadIdx.x/128)*16;
    float v[16];
    receipt(base+col,v);
    #pragma unroll
    for(int p=0;p<4;++p) {
      float g=v[4*p],h=v[4*p+2];
      float x=__fdividef(g,1.0f+__expf(-g))*v[4*p+1];
      float y=__fdividef(h,1.0f+__expf(-h))*v[4*p+3];
      reinterpret_cast<bf16_2*>(stage)[row*(Width/2)+((col/4+p)^(row&31))]
        =__floats2bfloat162_rn(x,y);
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  if constexpr(Threads==256) __syncthreads();
  else warpgroup::sync(1);
  auto* dst=static_cast<bf16*>(output);
  for(int i=threadIdx.x;i<128*Width/8;i+=Threads) {
    int r=i/(Width/8), c=(i%(Width/8))*8;
    const auto* words=reinterpret_cast<uint32_t*>(stage)+r*(Width/2);
    uint4 value{words[((c/2)+0)^(r&31)],words[((c/2)+1)^(r&31)],
                words[((c/2)+2)^(r&31)],words[((c/2)+3)^(r&31)]};
    *reinterpret_cast<uint4*>(dst+(args.m0+r)*(args.N/2)+args.n0/2+c)=value;
  }
  if constexpr(Threads==256) __syncthreads();
  else warpgroup::sync(1);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
// Stream the unfused bf16 result through two dead-operand slabs. Keeping
// only one 32-column packet in registers avoids a full-tile store tail.
__device__ inline void gemm_output(uint32_t base, coda_tile_args const& args,
                                   CUtensorMap const* map, void* memory) {
  if (threadIdx.x < 128) {
    auto* tiles = reinterpret_cast<st_bf<128,32>*>(memory);
    Accumulator acc(base);
    rt_bf<32,32> packet;
    #pragma unroll
    for (int chunk=0; chunk<8; ++chunk) {
      warpgroup::load_async(packet, acc.subtile<tt<float,128,32>>(0, chunk*32));
      tensor_load_wait();
      warpgroup::tma::store_async_read_wait<1>();
      warpgroup::sync(1);
      warpgroup::store(tiles[chunk%2], packet);
      warpgroup::sync(1);
      if (threadIdx.x == 0) {
        uint32_t address = uint32_t(__cvta_generic_to_shared(&tiles[chunk%2]));
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint "
                     "[%0, {%2,%3}], [%1], %4;"
                     :: "l"(map), "r"(address), "r"(args.n0+chunk*32), "r"(args.m0),
                        "l"(make_cache_policy<cache_policy::EVICT_FIRST>()) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      }
    }
    warpgroup::tma::store_async_read_wait<0>();
  }
  __syncthreads();
}
template<bool Fused>
__device__ inline void oneshot(NativeArgs const& p, unsigned char* shared,
                               CUtensorMap const* output_map=nullptr) {
  auto& op=*reinterpret_cast<Operands*>(shared);
  auto& s=*reinterpret_cast<State*>(shared+sizeof(Operands));
  coda_tile_args args{nullptr,nullptr,p.M,p.N,p.K,0,0};
  uint32_t base=produce(p.producer,args,op,s);
  if constexpr (Fused)
    epilogue<256>(base,args,p.output,reinterpret_cast<bf16*>(shared));
  else
    gemm_output(base,args,output_map,shared);
  release<false>(s);
}
}
extern "C" __global__ __launch_bounds__(256,2)
void coda_sm103_fused(const __grid_constant__ coda_sm103::NativeArgs p) {
  extern __shared__ __align__(1024) unsigned char native_shared[];
  coda_sm103::oneshot<true>(p,native_shared);
}
extern "C" __global__ __launch_bounds__(256,2)
void coda_sm103_unfused_gemm(const __grid_constant__ coda_sm103::GemmNativeArgs p) {
  extern __shared__ __align__(1024) unsigned char native_shared[];
  coda_sm103::oneshot<false>(p.main,native_shared,&p.d);
}
